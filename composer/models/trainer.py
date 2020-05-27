import math
import logging
from tqdm import tqdm
import tensorflow as tf
from composer.utils import merge_dicts
from composer.models import ModelSaveFrequencyMode

class Trainer:
    '''
    A generic model trainer.

    '''

    def __init__(self, model, logdir, restore_model=True, max_checkpoints=5, optimizer='adam',
                 learning_rate=1e-3, adam_epsilon=1e-8, optimizer_config=None, loss='SparseCategoricalCrossentropy',
                 loss_from_logits=True, loss_config=None, force_cpu=False):
        '''
        Initializes an instance of :class:`Trainer`.

        :param model:
            The :class:`composer.models.BaseModel` to train, evaluate, and predict with.
        :param logdir:
            The directory where checkpoints and TensorBoard summaries are saved.
        :param restore_model:
            Indicates whether to restore the model from the latest checkpoint in ``logdir``.
            Defaults to ``True``.
        :param max_checkpoints:
            The maximum number of checkpoints to keep. Defaults to 5.
        :param optimizer:
            The name of a TensorFlow otimizer class. Defaults to the Adam optimizer.
            See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers for a full
            list of Keras optimizer classes.
        :param learning_rate:
            The initial learning rate of the optimizer. Defaults to 1e-3.
        :param adam_epsilon:
            The epsilon for an Adam optimizer. Defaults to 1e-8.
        :param optimizer_config:
            A dictionary of keyword arguments to the optimizer object constructor.
        :param loss:
            The name of a TensorFlow loss class. Defaults to sparse categorical crossentropy.
            See https://www.tensorflow.org/api_docs/python/tf/keras/losses for a full
            list of Keras loss classes.
        :param loss_from_logits:
            Indicates whether the inputs to the loss object are in the form of logits.
            Defaults to ``True``.
        :param loss_config:
            A dictionary of keyword arguments to the loss object constructor.
        :param force_cpu:
            Indiciates whether to use the CPU even when CUDA is available.
            Defaults to ``False``.

        '''

        self.model = model
        self.logdir = logdir
        self.force_cpu = force_cpu
        self.max_checkpoints = max_checkpoints

        # Initialize the strategy
        gpus = tf.config.list_physical_devices('GPU')
        self.n_gpus = len(gpus)
        if self.force_cpu or self.n_gpus == 0:
            self.strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
        elif self.n_gpus == 1:
            self.strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
        elif self.n_gpus > 1:
            self.strategy = tf.distribute.MirroredStrategy()
        else:
            raise ValueError('Unable to resolve device strategy for training.')

        with self.strategy.scope():
            # Create optimizer and loss objects
            self.optimizer = tf.keras.optimizers.get({
                'class_name': optimizer,
                'config': merge_dicts({
                    'learning_rate': learning_rate,
                    'epsilon': adam_epsilon
                }, optimizer_config)
            })

            self.loss = tf.keras.losses.get({
                'class_name': loss,
                'config': merge_dicts({
                    'from_logits': loss_from_logits,
                    'reduction': tf.keras.losses.Reduction.NONE
                }, loss_config)
            })

            # Create the checkpoint manager
            self.checkpoint = tf.train.Checkpoint(
                step=tf.Variable(1), epoch=tf.Variable(1),
                optimizer=self.optimizer, model=self.model
            )

            self.checkpoint_manager = tf.train.CheckpointManager(
                self.checkpoint, self.logdir,
                max_to_keep=self.max_checkpoints
            )

            # Restore the model, if exists
            if restore_model:
                try:
                    checkpoint_path = self.checkpoint_manager.latest_checkpoint
                    self.checkpoint.restore(checkpoint_path).expect_partial()
                    logging.info('{} model restored from \'{}\' (global_step={}, epoch={}).'.format(
                        self.model.__class__.__name__, checkpoint_path,
                        self.checkpoint.step.numpy(), self.checkpoint.epoch.numpy()
                    ))
                except:
                    logging.exception('Failed to restore {} model from \'{}\''.format(
                        self.model.__class__.__name__, self.logdir
                    ))
                    
                    exit(1)

            # Initialize the summary writer.
            self.train_summary_writer = tf.summary.create_file_writer(str(logdir / 'train'))

    @tf.function
    def _train_step(self, x, y):
        '''
        A single training step.

        '''

        # Compute loss and optimize
        with tf.GradientTape() as tape:
            predictions, _ = self.model(x, training=True)
            loss = self.loss(y_true=y, y_pred=predictions)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return predictions, loss

    def train(self, dataset, epochs=None, save_frequency_mode=ModelSaveFrequencyMode.EPOCH,
              save_frequency=1, show_progress_bar=True):
        '''
        Trains the model.

        :param dataset:
            A :class:`tf.data.Dataset` object containing the dataset to train with.
        :param epochs:
            The number of epochs to train for. Defaults to ``None``, meaning
            that the model will train indefinitely.
        :param save_frequency_mode:
            A :class:`composer.models.ModelSaveFrequency` indicating the units of 
            the model save frequency. This can also be a string value corresponding
            to the enum value. Defaults to :class:`composer.ModelSaveFrequencyMode.EPOCH`.
        :param save_frequency:
            How often the model should be saved in units specified by the 
            `save_frequency_mode` parameter. Defaults to 1.
        :param show_progress_bar:
            Indicates whether a progress bar will be shown to indicate epoch status.
            Defaults to ``True``.

        '''

        steps_per_epoch = None
        save_frequency_mode = ModelSaveFrequencyMode(save_frequency_mode)
        while epochs is None or int(self.checkpoint.epoch) < epochs:
            current_epoch = int(self.checkpoint.epoch)
            logging.info('Epoch {}'.format(current_epoch if epochs is None else '{}/{}'.format(current_epoch, epochs)))
            with tqdm(total=steps_per_epoch, disable=not show_progress_bar) as progress_bar:
                epoch_loss_average = tf.keras.metrics.Mean()
                epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

                for x, y in dataset:
                    predictions, loss = self.strategy.reduce(
                        tf.distribute.ReduceOp.MEAN,
                        self.strategy.experimental_run_v2(
                            self._train_step,
                            args=(x, y)
                        ),
                        axis=None
                    )

                    loss = tf.reduce_mean(loss)

                    # Calculate the batch accuracy
                    _acc_pred_y = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
                    _acc_y = tf.cast(y, tf.int32)
                    accuracy = tf.reduce_mean(tf.cast(tf.equal(_acc_pred_y, _acc_y), tf.float32))

                    # Update loss and accuracy metrics
                    epoch_loss_average.update_state(loss)
                    epoch_accuracy.update_state(y, predictions)

                    # Log to TensorBoard summary
                    global_step = int(self.checkpoint.step)
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss, step=global_step)
                        tf.summary.scalar('accuracy', accuracy, step=global_step)

                    # Update description of progress bar to show loss and accuracy statistics
                    progress_bar.set_description('- loss: {:.4f} - accuracy: {:.4f}'.format(loss, accuracy))

                    if save_frequency_mode == ModelSaveFrequencyMode.GLOBAL_STEP and global_step % save_frequency == 0:
                        save_path = self.checkpoint_manager.save()
                        progress_bar.write('Saved checkpoint for step {} at {}.'.format(global_step, save_path))

                    self.checkpoint.step.assign_add(1)
                    progress_bar.update(1)

                # Log to TensorBoard summary
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('epoch_loss', epoch_loss_average.result(), step=current_epoch)
                    tf.summary.scalar('epoch_accuracy', epoch_accuracy.result(), step=current_epoch)

                if save_frequency_mode == ModelSaveFrequencyMode.EPOCH and current_epoch % save_frequency == 0:
                    save_path = self.checkpoint_manager.save()
                    progress_bar.write('Saved checkpoint for epoch {} at {}.'.format(current_epoch, save_path))
                
                if steps_per_epoch is None:
                    steps_per_epoch = progress_bar.n

                self.checkpoint.epoch.assign_add(1)
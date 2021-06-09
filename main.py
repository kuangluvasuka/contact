import time
import yaml
from absl import app
import tensorflow as tf


def config_gpu():
  pass



def main():

  from data_utils.proteinnet_serializer import deserialize_proteinnet_sequence
  from data_utils.vocabs import PFAM_VOCAB
  from feeder import Feeder
  from models import ConvModel, Resnet
  from train import Train
  from evaluate_metric import evaluation_metrics

  gpus = tf.config.list_physical_devices('GPU')
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

  # gpus = tf.config.list_physical_devices('GPU')
  # tf.config.set_logical_device_configuration(gpus[0],
  #                                            [tf.config.LogicalDeviceConfiguration(memory_limit=10240),
  #                                             tf.config.LogicalDeviceConfiguration(memory_limit=10240)])
  # logical_gpus = tf.config.list_logical_devices('GPU')
  # print(len(logical_gpus), "Logical GPUs")

  # load params
  with open('./params.yaml') as f:
      params = yaml.safe_load(f)
  params['vocab_size'] = len(PFAM_VOCAB)
  #params['resume_training'] = True

  # Initializing model
  if len(logical_gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
  else:
    strategy = tf.distribute.get_strategy()

  # load pretrained embedding
  embedding = None
  if params['use_pretrain']:
    import pickle
    embedding = pickle.load(open(params['pretrained_file'], 'rb'))  

  feeder = Feeder(params['data_folder'],
                  params['batch_size'],
                  params['test_batch_size'],
                  params['bucket_boundaries'],
                  deserialize_proteinnet_sequence,
                  distribute_strategy=strategy,
                  pretrained_embedding=embedding,
                  shuffle=params['shuffle'],
                  max_protein_length=params['max_protein_length'])
  del embedding

  with strategy.scope():
    model = ConvModel(params)
    optimizer = tf.optimizers.Adam(learning_rate=params['learning_rate'])
    train_obj = Train(model, optimizer, strategy, params, params['summary_dir'], params['checkpoint_dir'])

    train_obj.build()
    current_epoch = 1
    if params['resume_training']:
      #TODO: add path
      current_epoch = train_obj.restore_checkpoint(ckpt_path=None)

    # train and valid
    print("Training started from Epoch: {}".format(current_epoch))
    for epoch in range(current_epoch, params['epochs'] + 1):
      start = time.time()

      train_loss = train_obj.run_train_epoch(feeder.train)

      valid_loss, preds, trues, lengths = train_obj.run_test_epoch(feeder.valid)

      results = evaluation_metrics(preds, trues, lengths)

      train_obj.summarize_metrics(results, epoch)

      print("Epoch: {} | train average loss: {:.3f} | time: {:.2f}s | valid average loss: {:.3f})".format(
          epoch, train_loss.numpy(), time.time() - start, valid_loss.numpy()))

      if epoch % params['checkpoint_inteval'] == 0:
        train_obj.save_checkpoint(epoch)


if __name__ == '__main__':
  app.run(main)

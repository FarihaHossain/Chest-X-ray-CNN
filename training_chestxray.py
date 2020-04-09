import argparse
import keras
import keras.backend as K
import gc
import time
from source.load_data import Chest_XRay
from source.model import spatial_feature_aggregation_block,spa_pooling,SPANet
from source.call_back import callback_for_training
from source.output_visualization import plot_loss_acc


def training_chestXray(data_dir,logdir,input_dim,dataset,batch,epoch,weights,snapshot_name):


	if dataset=='CXRay_pnu':
        train_batches, test_batches = Chest_XRay(batch_size, input_size, data_dir)
        num_of_classes = 2
        train_size = 5232
        test_size = 624
    # clear unnecessary memory GPU & RAM   
    K.clear_session()
    gc.collect()

    # Calculate the starting time
    
    start_time = time.time()

    # Callbacks for model saving, adaptive learning rate
    cb = callback_for_training(tf_log_dir_name=logdir,snapshot_name=snapshot_name)


    # Loading the model
    model = SPANet(img_size=input_size,num_cls=num_of_classes)

    # Training the model
    history = model.fit_generator(train_batches, shuffle=True, steps_per_epoch=train_size //batch_size, validation_data=test_batches, validation_steps= test_size//batch_size, epochs=epoch, verbose=1, callbacks=cb)


    end_time = time.time()

    print("--- Time taken to train : %s hours ---" % ((end_time - start_time)//3600))

    # Saving the final model
    if snapshot_name == None :
        model.save('SPANet.h5')
       
    else :    
        model.save(snapshot_name+'.h5')
    
    plot_loss_acc(history,snapshot_name)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='CPnu dataset', choices=['CXRay_pnu'])
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--input_dim', type=int, default=224)
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory path: ')
	parser.add_argument('--logdir', type=str)
	parser.add_argument('--epoch', type=int, default=30)
	parser.add_argument('--weights', type=str,default=None, help='Previous weights: Resume')
	parser.add_argument('--snapshot_name',type=str, default=None, help='Snapshot Name : Saved')	
	args = parser.parse_args()
	training_chestXray(args.data_dir, args.logdir, args.input_dim, args.dataset, args.batch, args.epoch, args.weights, args.snapshot_name)



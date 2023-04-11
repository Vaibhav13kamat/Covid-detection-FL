#server twiks


server_rounds=3
server_grpc_max_message_length = 1024
#server_address = 'localhost:'+str(sys.argv[1])  
server_address = 'localhost:9969'

##################################################################


#client1 twiks
client1_weights='imagenet'
include_top=False
input_shape=(224, 224, 3)
client1_number_of_classes=2

#dataset_dirs
client1_training_dir = '/workspaces/Covid-detection-FL/dataset_split/client1/train'
client1_testing_dir = '/workspaces/Covid-detection-FL/dataset_split/client1/test'
client1_batch_size = 32
client1_epochs=1
client1_verbose=0
client1_grpc_max_message_length=1024

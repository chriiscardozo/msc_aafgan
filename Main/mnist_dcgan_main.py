# Needs refactoring. Check the mnist_condgan_main.py
'''
import Util
import time,os
import torch
from Models.mnist_DCGAN import DCGAN
import torch
from compare_ll import compare as Compare_LL

def main():
    start = time.time()

    seed = 20191

    # DCGAN MNIST setup #
    #*********************************#
    BATCH_SIZE = 64
    EPOCHS = 50
    PIN_MEMORY = (Util.DEVICE != 'cpu')
    WORKERS = 8
    OUTPUT_PREFIX = 'output'
    NUM_TEST_SAMPLES = 25
    VERBOSE_BATCH_STEP = 100
    VERBOSE_EPOCH_STEP = 1
    N_SEEDS = 10
    IMG_SIZE = 28
    #*********************************#

    seeds = [3355234131,796230325,2081402747,3669639015,1730802958,3784911998,432945918,1701650610,2946186485,707892250]
    #seeds = Util.generate_seeds(N_SEEDS)
    print(seeds)
    #with open('seeds'+str(time.time()), 'w') as f:
    #    f.write(','.join([str(x) for x in seeds]))

    train_data = Util.get_mnist_data(img_size=IMG_SIZE)
    test_data = Util.get_mnist_data(train=False,img_size=IMG_SIZE)
    X_train = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY, num_workers=WORKERS)
    X_test = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=True, pin_memory=PIN_MEMORY, num_workers=WORKERS)
    X_test = next(iter(X_test))[0].view(10000, -1)

    batches = len(X_train)
    print("Running", EPOCHS, "epochs with", batches, "batches each / Batch size:", BATCH_SIZE)

    D_Models = ["default", "BHSA", "BHAA"]
    G_Models = ["default", "BHSA", "BHAA"]
    #D_Models = ["default"]
    #G_Models = ["default"]

    test_noise = Util.noise(NUM_TEST_SAMPLES)
    test_noise = test_noise.view(-1, 100, 1, 1)


    for d_model in D_Models:
        for g_model in G_Models:
            for d_shrelu in [False,True]:
                for g_shrelu in [False,True]:
                    output_dir = OUTPUT_PREFIX
                    if(d_shrelu): output_dir += "_Dshrelu"
                    if(g_shrelu): output_dir += "_Gshrelu"
                    if(d_model != "default"): output_dir += "_Dis_" + d_model
                    if(g_model != "default"): output_dir += "_Gen_" + g_model
                    
                    # temp
                    if(os.path.exists(output_dir)): 
                        continue
                    #

                    Util.reset_dir(output_dir)

                    try:
                        for i, seed in enumerate(seeds):
                        
                            output_dir_seed = os.path.join(output_dir, str(i))
                            Util.reset_dir(output_dir_seed)

                            torch.backends.cudnn.deterministic = True
                            torch.manual_seed(seed)
                            torch.cuda.manual_seed_all(seed)

                            dcgan = DCGAN(d_model=d_model,g_model=g_model,d_shrelu=d_shrelu,g_shrelu=g_shrelu,output_dir=output_dir_seed,img_size=IMG_SIZE)
                            dcgan.train(data_loader=X_train,X_test=X_test,epochs=EPOCHS,verbose_batch=VERBOSE_BATCH_STEP,verbose_epoch=VERBOSE_EPOCH_STEP,fixed_test_noise=test_noise)
                    except:
                            print("*** Exception ***")
                            f = open(os.path.join(output_dir,"exception"), "w")
                            f.close()
                        

    Compare_LL()
    Util.exec_time(start, "Running")

if __name__ == '__main__':
    main()
'''
from sinenet_torch_BN import *
from extract_patches import *
from prediction_sn import *
from test_retina_keras import *
import configparser as cfp


class Sine_t():

    def __init__(self):
        self.path_experiment = None
        self.test_FOVs = None
        self.test_masks = None
        self.test_original = None
        self.check_path_cont = None
        self.name_experiment = None

    def load_dataS(self):

        # path_drive = '../../../../DRIVE/DRIVE'
        path_drive = '../../data/DRIVE'
        image_folder = path_drive + '/test/images/'
        images = [img for img in os.listdir(image_folder) if img.endswith(".tif")]
        images = sorted(images, key=natural_sort_key)  # sorted

        gt_folder = path_drive + '/test/1st_manual/'
        gts = [img for img in os.listdir(gt_folder) if img.endswith(".gif")]
        gts = sorted(gts, key=natural_sort_key)  # sorted

        mask_folder = path_drive + '/test/mask/'
        masks = [img for img in os.listdir(mask_folder) if img.endswith(".gif")]
        masks = sorted(masks, key=natural_sort_key)  # sorted

        test_original, test_masks, test_FOVs = np.zeros(shape=(len(images), 584, 565), dtype=np.uint8), np.zeros(
            shape=(len(gts), 584, 565), dtype=np.uint8), np.zeros(shape=(len(masks), 584, 565), dtype=np.uint8)
        for i in range(len(images)):
            test_original[i] = cv2.imread(os.path.join(image_folder, images[i]), 0)
            gif_reader = imageio.get_reader(os.path.join(gt_folder, gts[i]))
            gif_length = gif_reader.get_length()
            frame_index = 0
            if frame_index < gif_length:
                test_masks[i] = gif_reader.get_data(frame_index)
            gif_reader = imageio.get_reader(os.path.join(mask_folder, masks[i]))
            gif_length = gif_reader.get_length()
            frame_index = 0
            if frame_index < gif_length:
                test_FOVs[i] = gif_reader.get_data(frame_index)

        config = cfp.RawConfigParser()
        config.read('./configuration.txt')
        # path to the datasets
        data_path = config.get('data paths', 'path_local')
        self.name_experiment = config.get('experiment name', 'name')
        self.path_experiment = './' + self.name_experiment + '/'
        # N full images to be predicted
        # Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))

        test_original = test_original[:, 10:575, :]
        test_masks = test_masks[:, 10:575, :]
        test_FOVs = test_FOVs[:, 10:575, :]
        del gts, images
        gc.collect()

        self.test_original = (test_original / 255.).astype(np.float32)
        self.test_masks = (test_masks / 255.).astype(np.float32)
        self.test_FOVs = (test_FOVs / 255.).astype(np.float32)
        print(self.test_FOVs.shape, self.test_original.shape, self.test_masks.shape)
        del test_original, test_masks, test_FOVs

    def Sinenet_predict(self):
        
        print('==> Sine-Net load Test data..')
        self.load_dataS()

        print('==> Sine-Net Test from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        self.check_path_cont = 'SineNet_layer_%d_filter_%d_continue_test_TverskyLoss.pt7'% (15,3)
        checkpoint = torch.load('./checkpoint/' + self.check_path_cont, map_location=torch.device('cpu'))
        model = Sine_Net(n_channels=1, n_classes=1)
        model.load_state_dict(checkpoint['model_state'])

        params = {
            "model_name": "Sine_Net",
            "device": 'cuda',  ### device
            "batch_size": 4,
            "num_workers": 2,
            "model": model
        }

        model = model.half().to(params["device"])
        ## Create test image's patches per image
        for i in range(self.test_original.shape[0]):
            eval = Test(params, 448, 448, self.test_original[i], self.test_masks[i], self.test_FOVs[i], self.path_experiment, i)
            pred_patches = eval.inference(model)
            print(eval.evaluate())
            eval.save_segmentation_result()
            
        return pred_patches
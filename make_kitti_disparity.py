from Test_img import *
import os

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015', help='KITTI version')
parser.add_argument('--loadmodel', default='./trained/pretrained_model_KITTI2015.tar', help='loading model')
parser.add_argument('--leftimg', default= './VO04_L.png', help='load model')
parser.add_argument('--rightimg', default= './VO04_R.png', help='load model')
parser.add_argument('--dataset-path', default= './dataset/KT15/', help='load model')
parser.add_argument('--results-path', default= './results/', help='load model')
parser.add_argument('--model', default='stackhourglass', help='select model')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
	model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
	model = basic(args.maxdisp)
else:
	print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
	print('load PSMNet')
	state_dict = torch.load(args.loadmodel)
	model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


if __name__ == '__main__':
	results_path = args.results_path
	if args.KITTI in ['15', '2015']:
		training_left_path = os.path.join(args.dataset_path, 'training/image_2/')
		training_right_path = os.path.join(args.dataset_path, 'training/image_3/')
		for i in range(200):
			filename = str(i).zfill(6) + '_10.png'
			args.leftimg = os.path.join(training_left_path, filename)
			args.rightimg = os.path.join(training_right_path, filename)
			args.results = os.path.join(results_path, 'KT15/training/')
			Test(args, model)
		testing_left_path = os.path.join(args.dataset_path, 'testing/image_2/')
		testing_right_path = os.path.join(args.dataset_path, 'testing/image_3/')
		for i in range(200):
			filename = str(i).zfill(6) + '_10.png'
			args.leftimg = os.path.join(testing_left_path, filename)
			args.rightimg = os.path.join(testing_right_path, filename)
			args.results = os.path.join(results_path, 'KT15/testing/')
			Test(args, model)

	if args.KITTI in ['12', '2012']:
		training_left_path = os.path.join(args.dataset_path, 'training/image_0/')
		training_right_path = os.path.join(args.dataset_path, 'training/image_1/')
		for i in range(194):
			filename = str(i).zfill(6) + '_10.png'
			args.leftimg = os.path.join(training_left_path, filename)
			args.rightimg = os.path.join(training_right_path, filename)
			args.results = os.path.join(results_path, 'KT12/training/')
			Test(args, model)
		testing_left_path = os.path.join(args.dataset_path, 'testing/image_0/')
		testing_right_path = os.path.join(args.dataset_path, 'testing/image_1/')
		for i in range(195):
			filename = str(i).zfill(6) + '_10.png'
			args.leftimg = os.path.join(testing_left_path, filename)
			args.rightimg = os.path.join(testing_right_path, filename)
			args.results = os.path.join(results_path, 'KT12/testing/')
			Test(args, model)

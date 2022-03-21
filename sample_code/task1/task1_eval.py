import cv2 
import os
import numpy as np 
import argparse

def main(args):
	gt_dir = args.gt_dir 
	pred_dir  = args.pred_dir 

	img_path_list = os.listdir(gt_dir)
	precision_list = []
	recall_list = []

	for img_path in img_path_list:
		gt_path = os.path.join(gt_dir, img_path)
		pred_path = os.path.join(pred_dir, img_path)

		gt_img = cv2.imread(gt_path)[:,:,0] == 255
		pred_img = cv2.imread(pred_path)[:,:,0] > 127

		tp = np.sum(np.logical_and(pred_img, gt_img))
		gt = np.sum(gt_img)
		pred = np.sum(pred_img)

		if pred !=0 and gt!=0:
			precision = 1.0 * tp/pred 
			recall = 1.0 * tp / gt

			precision_list.append(precision)
			recall_list.append(recall)

		#print(precision,recall )
	
	print('Average precision', np.mean(precision_list))
	print('Average recall', np.mean(recall_list))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, default='../../data/task1_data/task1_test/mask',
                        help='the ground-truth mask directory.')
    parser.add_argument('--pred_dir', type=str, default='task1_out',
                        help='the output folder to save all the output probability maps')
    
    args = parser.parse_args()
    print(args)

    
    main(args)

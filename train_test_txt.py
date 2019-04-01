import os
import sys

def main():
	path = "/Users/weixingxu/Desktop/Formate_transformation/bdd2yolo3/yolo3/labels/train" 
	files = [p for p in os.listdir(path) if '.' != p[0]]
	files.sort()
	files = files[0:15]
	# print(files)
	# sys.exit(-1)

	with open('train.txt','w') as fout_train, open('test.txt','w') as fout_test:
		for file in files[:10]:
			filename = file[:-4]
			print (filename)	
			
			if os.path.isdir(file) or 'txt' not in file:
				continue

			fout_train.write(filename + ' ')			

		for file in files[10:]:
			filename = file[:-4]
			print (filename)
			
			if os.path.isdir(file) or 'txt' not in file:
				continue
			
			fout_test.write(filename + ' ')	


if __name__ == '__main__':
	main()
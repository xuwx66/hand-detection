import os
import sys

def main():
	path = "/Users/weixingxu/Desktop/Formate_transformation/bdd2yolo3/yolo3/labels/val" 
	files = [p for p in os.listdir(path) if '.' != p[0]]
	files.sort()
	files = files[0:15]
	# print(files)
	# sys.exit(-1)

	with open('trainval.txt','w') as fout_trainval, open('val.txt','w') as fout_val:
		for file in files[:10]:
			filename = file[:-4]
			print (filename)	
			
			if os.path.isdir(file) or 'txt' not in file:
				continue

			fout_trainval.write(filename + ' ')			

		for file in files[10:]:
			filename = file[:-4]
			print (filename)
			
			if os.path.isdir(file) or 'txt' not in file:
				continue
			
			fout_val.write(filename + ' ')	

if __name__ == '__main__':
	main()
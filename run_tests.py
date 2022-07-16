# This script is meant to carry out an automated test on all examples
# of Nabla and log the console output of each to a text file which is
# created and placed in the same directory as the executable of each example


import glob, os, subprocess


# walk into the examples_tests directory
os.chdir("examples_tests")

# list containing indices of each example to test
# (I'm supposed to use list comprehension here but for some weird reason 'glob' refuse to work when I use it :\)
indices = [1, 2, 3] 	# This list should contain indices of all examples to be tested

# format each element of 'indices' to 2-digit
def format_list():
	for i in range(len(indices)):
		indices[i] = str(indices[i])
		indices[i] = '0' + indices[i]
		if len(indices[i]) > 2:
			indices[i] = indices[i].replace(indices[i][0], '', 1)


# Deduce the exact filename of example to be tested
def strip_name(filename):
	for i in range(len(filename)):
		if filename[i] == '.':
			filename = filename[i+1:]
			break
	filename = filename + ".exe"

	return filename.lower()


def run_tests():
	for i in range(len(indices)):
		for fn in glob.glob(indices[i] + ".*"):
			print("\n", fn)
			filename = strip_name(fn)
			print('',filename)
			os.chdir(fn)

			# if "bin" in list(filter(os.path.isdir, os.listdir(os.curdir))):
			try:
				os.chdir("bin")
				p = subprocess.Popen(filename, stdout=subprocess.PIPE, shell=False, text=True)
				outs, errs = p.communicate()

				if p.returncode == 0:
					return_status = 'SUCCESS'
				else:
					return_status = 'FAILURE'
				print('', return_status, '\n', 'EXIT STATUS:', p.returncode)

				# create a text file, and log exit code and console output
				with open("log.txt", mode='w', encoding='utf-8') as log_file:
					log_file.write('\n EXIT STATUS: ' + str(p.returncode) + '\n\n\n\nCONSOLE OUTPUT:\n\n')
					log_file.write(outs)
				os.chdir("../..")
			except FileNotFoundError:
			# else:
				return_status = 'FAILURE'
				print('', return_status, '\n', '\'bin\' directory not found')
				with open("log.txt", mode='w', encoding='utf-8') as log_file:
					log_file.write('\n ' + return_status + '\n ' + '\'bin\' directory not found')
				os.chdir("..")



if __name__ == '__main__':
	format_list()
	run_tests()

	input("\n DONE! PRESS ENTER TO CLOSE ")


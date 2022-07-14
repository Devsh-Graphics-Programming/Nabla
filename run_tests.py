# This script is meant to carry out an automated test on all examples
# of Nabla and log the console output of each to a text file which is
# created and placed in the same directory as the executable of each example


import glob, os, subprocess, time


# walk into the examples_tests directory
os.chdir("examples_tests")

# list containing indices of each example to test
# (I'm supposed to use list comprehension here but for some weird reason 'glob' refuse to work when I use it :\)
indices = [1,2,3]

# format each element of 'indices' to 2 digit
def format_list():
	for i in range(len(indices)):
		indices[i] = str(indices[i])
		indices[i] = '0' + indices[i]
		if len(indices[i]) > 2:
			indices[i] = indices[i].replace(indices[i][0], '', 1)


def run_tests():
	for i in range(len(indices)):
		for fn in glob.glob(indices[i] + ".*"):
		    print("\n", fn)
		    os.chdir(fn)
		    os.chdir("bin")

		for fn in glob.glob("*.exe"):
			p = subprocess.Popen(fn, stdout=subprocess.PIPE, shell=False, text=True)
			try:
				outs, errs = p.communicate(timeout=10)
			except subprocess.TimeoutExpired:
				p.terminate()
				p.wait()
				outs, errs = p.communicate()

				if p.returncode == 0 or p.returncode == 1:
					return_status = 'SUCCESS'
				else:
					return_status = 'FAILURE'
				print('EXIT STATUS CODE: ', p.returncode, '(', return_status, ')')

				# create a text file, and log exit code and console output
				with open("log.txt", mode='w', encoding='utf-8') as log_file:
					log_file.write('\nEXIT STATUS CODE: ' + str(p.returncode) + '\n\n\n\nCONSOLE OUTPUT:\n\n')
					log_file.write(outs)

		os.chdir("../..")


if __name__ == '__main__':
	format_list()
	run_tests()


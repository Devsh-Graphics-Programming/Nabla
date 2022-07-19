# This script is meant to carry out an automated test on all examples
# of Nabla and log the console output of each to a text file which is
# created and placed in the same directory as the executable of each example


import glob, os, subprocess, re


# list containing indices of each example to test
# (I'm supposed to use list comprehension here but for some weird reason 'glob' refuse to work when I use it :\)
indices = [0, 1, 2, 3, 9, 34] 	# This list should contain indices of all examples to be tested


html_head = """
<html>
	<head> <title>TEST RESULTS</title> </head>
	<style type="text/css">
		table, th, td
		{
			border:1px solid black;
		}
	</style>
	<body>
		<br>
		<h2>TEST RESULTS</h2>
		<table >
			<tr style="background-color: GREY; color: white;">
				<th>EXAMPLE</th>
				<th>STATUS</th>
				<th>STATUS CODE</th>
				<th>MESSAGE</th>
			</tr>
"""

def init_html():
	with open("test_results.html", mode='w') as html_file:
		html_file.write(html_head)
		html_file.close()

def html_foot():
	with open("test_results.html", mode='a') as html_file:
		html_file.write("\t\t</table>\n")
		html_file.write("\t</body>\n")
		html_file.write("</html>\n")
		html_file.close()



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

def fill_row(example_name, return_status, status_code, message):
	with open("test_results.html", mode='a') as html_file:
		html_file.write("\t\t\t<tr>\n")
		html_file.write("\t\t\t\t<td>" + example_name + "</td>\n")
		if return_status == "SUCCESS":
			html_file.write("\t\t\t\t<td style=\"background-color: green; color: white\">" + return_status + "</td>\n")
		else:
			html_file.write("\t\t\t\t<td style=\"background-color: red; color: white\">" + return_status + "</td>\n")
		html_file.write("\t\t\t\t<td>" + str(status_code) + "</td>\n")
		html_file.write("\t\t\t\t<td>" + message + "</td>\n")
		html_file.write("\t\t\t</tr>\n")
		html_file.close()


def handle_validation_errors(data):
	pattern = re.compile(r"Validation Error:[\s\S]+\n")
	matches = pattern.finditer(data)

	validation_errors = ""
	for match in matches:
		validation_errors += match.group()

	return str(validation_errors).split("\n")


def run_tests():
	for i in range(len(indices)):
		for fn in glob.glob(indices[i] + ".*"):
			print("\n", fn)
			filename = strip_name(fn)
			print('',filename)
			os.chdir(fn)
			returncode = ""

			if "bin" in list(filter(os.path.isdir, os.listdir(os.curdir))):
				os.chdir("bin")
				if os.path.exists(filename):
					p = subprocess.Popen(filename, stdout=subprocess.PIPE, shell=False, text=True)
					try:
						outs, errs = p.communicate(timeout=10)
					except subprocess.TimeoutExpired:
						p.terminate()
						p.wait()
						outs, errs = p.communicate()

					if p.returncode == 0 or p.returncode == 1:
						returncode = "0"
						return_status = 'SUCCESS'
						message = "Succesful execution"
						if handle_validation_errors(outs) != [""]:
							return_status = "FAILURE"
							message += " with Validation Errors"
					else:
						returncode = str(p.returncode)
						return_status = 'FAILURE'
						message = "Application failed (" + str(hex(p.returncode)) + ")"
					print('', return_status, '\n', 'EXIT STATUS:', returncode)

					# create a text file, and log exit code and console output
					with open("log.txt", mode='w', encoding='utf-8') as log_file:
						log_file.write('\n EXIT STATUS: ' + str(p.returncode) + '\n\n\n\nCONSOLE OUTPUT:\n\n')
						log_file.write(outs)
					os.chdir("../..")
					fill_row(fn, return_status, p.returncode, message)
				else:
					returncode = "NIL"
					return_status = 'FAILURE'
					error_message = 'Release build not found'
					print('', return_status, '\n', error_message)
					with open("log.txt", mode='w', encoding='utf-8') as log_file:
						log_file.write('\n ' + return_status + '\n ' + error_message)
					os.chdir("../..")
					fill_row(fn, return_status, returncode, error_message)
			else:
				returncode = "NIL"
				return_status = 'FAILURE'
				error_message = '\'bin\' directory not found'
				print('', return_status, '\n', error_message)
				with open("log.txt", mode='w', encoding='utf-8') as log_file:
					log_file.write('\n ' + return_status + '\n ' + error_message)
				os.chdir("..")
				fill_row(fn, return_status, "1", error_message)
	html_foot()



if __name__ == '__main__':

	# walk into the examples_tests directory
	os.chdir("examples_tests")

	init_html()
	format_list()
	run_tests()

	input("\n DONE! PRESS ENTER TO CLOSE ")

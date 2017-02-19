#This file provide all input functions for the machine.

import sys
import Msupport
import time
import re

def input():
	while 1:
		print "Please give me your new order!"
		order = input_try()
		if order == 0:
			continue
		print "Here is the order you gave me:"
		print "Number of orders: ", order[0]
		for i in range(1,order[0]+1):
			Msupport.dictprint(order[i])
		print "Are these orders correct?"
		request = requestcheck()
		if request:
			return order
		


def input_try():
	order = raw_input()
	
	order_organized = list()
	order_organized.append(0)
	
	if "f:" in order:
		try:
			input_filename = re.findall('f:(\S+)',order)
			#print input_filename
			input_fo = open(input_filename[0],'r')
			print input_filename[0], " successfully loaded."
			for line in input_fo.readlines():
				if line.startswith('#') or not line.split():
					continue
				else:
					order_organized.append(input_readline(line))
					order_organized[0] = order_organized[0] + 1
			#input_file = input_fo.read()
		except:
			print "Failed read orders from ", input_filename 
			return 0
			
	else:
		order_organized.append(input_readline(order))
		order_organized[0] = order_organized[0] + 1
		
		
	return order_organized
	
def input_readline(line):
	dict = {
		"time": time.asctime( time.localtime(time.time()) ),
		"game": None,
		"job": None,
		"machine": None,
		"para": None
	}
	
	order_dict = {
		"job": "-j",
		"machine": "-m",
		"para": "-p"
	}
	
	try:
		cur_order = re.findall('^(\S+) ',line)
		dict["game"] = cur_order[0]
	except:
		pass
	
	for key, value in order_dict.items():
		dict[key] = para_extract(order_dict,key,line)
	
	return dict
	
def para_extract(order_dict,str,line):
	para = None
	if order_dict[str] in line:
		para = list()
		para.append(True)
		try:
			cur_order = re.findall(order_dict[str]+'(.*?)(?:-|$)',line)
			for i in cur_order:
				para = para + i.split()
		except:
			pass
	return para


def requestcheck():
	print "(Y for yes, N for no, Q for quit)"
	order = raw_input()
	if order == "Q":
		sys.exit(0)
	elif order == "Y":
		return 1
	elif order == "N":
		return 0
	else:
		return requestcheck()
	
		

	

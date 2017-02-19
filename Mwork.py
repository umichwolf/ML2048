# This is how the machine is going to work due to the input.
import sys
import random
import re

import Msupport
import Moutput
import game_2048



def _2048_work(order):
	game = game_2048.Game_play(order)
	
	job_dict = {
		"oneplay": game.oneplay,
		"play": game.play,
		"playselect": game.playselect
		}
	para_dict = {
		"method" : None,
		"outputfile" : None
		}
		
	print Msupport.parametrize(order["para"])
	
	(job_dict[(order["job"])[1]])(Msupport.parametrize(order["para"]))
	return 0

def work(order):
	game_dict = {
		"2048": _2048_work
	}
	for i in range(1,order[0]+1):
		print "Start processing order ",i
		try:
			(game_dict[(order[i])["game"]])(order[i])
			print "Order process complete."
		except:
			print "Failed to process the following order:"
			Msupport.dictprint(order[i])
	
	print "Job done!"
	return 0
	
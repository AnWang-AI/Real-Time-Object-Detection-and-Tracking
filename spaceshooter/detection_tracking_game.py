from __future__ import division

import multiprocessing

import sys

from detection_and_tracking import get_object_center

from simulate_game import space_shooter

sys.path.append("..")

q = multiprocessing.Queue()

# classes:
# 1.Aeroplanes     2.Bicycles   3.Birds       4.Boats           5.Bottles
# 6.Buses          7.Cars       8.Cats        9.Chairs          10.Cows
# 11.Dining tables 12.Dogs      13.Horses     14.Motorbikes     15.People
# 16.Potted plants 17.Sheep     18.Sofas      19.Trains         20.TV/Monitors

# 指定进行检测和跟踪的类型
OBJECT_CLASS = 5

set_process = multiprocessing.Process(target=get_object_center, args=(q, OBJECT_CLASS,))
game_process = multiprocessing.Process(target=space_shooter, args=(q,))

set_process.start()
game_process.start()

game_process.join()
set_process.terminate()

print("退出主线程")

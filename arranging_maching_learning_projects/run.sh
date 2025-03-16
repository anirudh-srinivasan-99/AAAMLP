#!/bin/sh
python3 -m arranging_maching_learning_projects.src.train --fold 0 --model rf
python3 -m arranging_maching_learning_projects.src.train --fold 1 --model rf
python3 -m arranging_maching_learning_projects.src.train --fold 2 --model rf
python3 -m arranging_maching_learning_projects.src.train --fold 3 --model rf
python3 -m arranging_maching_learning_projects.src.train --fold 4 --model rf
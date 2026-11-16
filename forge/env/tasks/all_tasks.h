#pragma once
// Every task, for the name lookup in make_env. Add a new task here.
#include "forge/env/tasks/ant.h"
#include "forge/env/tasks/cartpole.h"
#include "forge/env/tasks/g1_stand.h"
#include "forge/env/tasks/g1_walk.h"
#include "forge/env/tasks/go2_walk.h"
#include "forge/env/tasks/panda_pick.h"

#define FORGE_FOR_EACH_TASK(X) X(CartpoleTask) X(AntTask) X(G1StandTask) X(G1WalkTask) X(G1WalkRoughTask) \
    X(Go2WalkTask) X(Go2WalkRoughTask) X(PandaPickTask) X(PandaPickColorTask)

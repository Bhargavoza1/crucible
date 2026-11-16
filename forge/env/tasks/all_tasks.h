#pragma once
// Every task, for the name lookup in make_env. Add a new task here.
#include "forge/env/tasks/ant.h"
#include "forge/env/tasks/cartpole.h"
#include "forge/env/tasks/g1_stand.h"

#define FORGE_FOR_EACH_TASK(X) X(CartpoleTask) X(AntTask) X(G1StandTask)

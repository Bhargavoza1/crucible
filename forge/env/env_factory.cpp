// make_env: picks the backend. In forge (not forge_cpu), so it can see CUDA.
#include <stdexcept>

#include "forge/env/env.h"

namespace forge {

std::unique_ptr<Env> make_env(const EnvOptions& opt) {
    if (opt.backend == EnvBackend::kCpu) return make_env_cpu(opt);
#ifdef CRUCIBLE_HAS_CUDA
    return make_env_gpu(opt);
#else
    throw std::runtime_error("backend gpu needs a build with CUDA (use backend cpu)");
#endif
}

}  // namespace forge

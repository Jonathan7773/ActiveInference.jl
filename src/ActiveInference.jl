module ActiveInference

include("maths.jl")
include("Environments\\EpistChainEnv.jl")
include("agent.jl")
include("utils.jl")
include("inference.jl")
include("learning.jl")

export # maths.jl
       norm_dist,
       sample_category,
       softmax,
       spm_log_single,
       entropy_A,
       kl_divergence,
       get_joint_likelihood,
       dot_likelihood,
       spm_log_array_any,
       softmax_array,
       spm_cross,
       spm_cross_learning,
       spm_dot,
       spm_MDP_G,
       norm_dist_array,
       spm_wnorm,


       # utils.jl
       array_of_any, 
       array_of_any_zeros, 
       array_of_any_uniform, 
       onehot,
       construct_policies_full,
       plot_gridworld,
       process_observation,
       get_model_dimensions,
       to_array_of_any,
       select_highest,
       action_select,


       # agent.jl
       initialize_agent,
       infer_states!,
       infer_policies!,
       sample_action!,
       update_A!,
       update_B!,
       update_D!,

       # inference.jl
       get_expected_states,
       update_posterior_states,
       fixed_point_iteration,
       compute_accuracy,
       calc_free_energy,
       update_posterior_policies,
       get_expected_obs,
       calc_expected_utility,
       calc_states_info_gain,
       calc_pA_info_gain,
       calc_pB_info_gain
       sample_action,

       # learning.jl
       update_obs_likelihood_dirichlet,
       update_state_likelihood_dirichlet,
       update_state_prior_dirichlet


    # From Environments\\EpistChainEnv.jl
    module Environments

    include("Environments\\EpistChainEnv.jl")
    
    export EpistChainEnv, step!, reset!
       
    end
end







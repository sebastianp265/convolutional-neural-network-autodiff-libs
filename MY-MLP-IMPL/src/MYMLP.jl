module MYMLP

include("network.jl")
export Dense, Chain, relu

include("computional-graph.jl")
export GraphNode, Operator, Constant, Variable,
    ScalarOperator, BroadcastedOperator, topological_sort, compute!,
    gradient, evaluate!

end

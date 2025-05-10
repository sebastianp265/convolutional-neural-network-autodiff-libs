relu(x) = ifelse(x < 0, zero(x), x)
sigmoid(x) = begin
    t = exp(-abs(x))
    ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
end

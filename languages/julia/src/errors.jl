struct AgentBibleError <: Exception
    message::String
end

Base.showerror(io::IO, err::AgentBibleError) = print(io, err.message)

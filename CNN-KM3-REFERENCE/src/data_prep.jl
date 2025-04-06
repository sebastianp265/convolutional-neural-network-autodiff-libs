using JLD2
using TextAnalysis, Languages
using Random

function prepare_dataset(n_samples=50_000, split_ratio=0.8, max_len=130)
	@info "Loading raw dataset..."
	reviews = load("data/imdb_dataset.jld2", "reviews")[1:n_samples]
	labels = load("data/imdb_dataset.jld2", "labels")[1:n_samples]

	indices = shuffle(1:n_samples)
	reviews = reviews[indices,:]
	labels = labels[indices]

	split_ratio = split_ratio
	split_index = Int(floor(split_ratio * n_samples))

	reviews_train = reviews[1:split_index]
	labels_train = labels[1:split_index]
	reviews_test = reviews[split_index+1:end]
	labels_test = labels[split_index+1:end]

	@info "Loading Glove embeddings..."
	embeddings = load("data/glove_6B_50d.jld2","embeddings");
	embedding_dim = length(first(embeddings)[2]);

	@info "Data preparation..."
	crps = Corpus(StringDocument.(reviews_train))
	languages!(crps, Languages.English())

	remove_case!(crps)
	remove_html_tags!(crps)
	remove_patterns!(crps, r"https?://[^\s]+|www\.[^\s]+")
	prepare!(crps, strip_pronouns)
	prepare!(crps, strip_articles)
	prepare!(crps, strip_prepositions)
	prepare!(crps, strip_stopwords)
	prepare!(crps, strip_punctuation)
	prepare!(crps, strip_whitespace)
	prepare!(crps, strip_numbers)

	update_lexicon!(crps)
	lex = lexicon(crps)

	# replace words with frequency less than min_freq woth oov word
	min_freq = 30
	frequent_words = keys(filter(p->p[2] >= min_freq, lex))
	oov_word = "<unk>"

	for i in 1:length(crps)
	    crps[i].text = join(map(x->x in frequent_words ? x : oov_word, split(crps[i].text))," ")
	end

	update_lexicon!(crps)
	lex = lexicon(crps)

	# get vocabulary
	vocab = collect(keys(lex))
	push!(vocab, "<pad>") # padding

	# tokenize reviews
	tokenized_train = map(x->split(text(x)), crps)
	tokenized_test = map(x->split(x), reviews_test)

	# replace unknown words in testing data
	tokenized_test = map(review->replace!(x->x in vocab ? x : oov_word, review), tokenized_test)

	# unknown and padding tokens get zeroed embedding
	embeddings["<unk>"] = zeros(Float32, embedding_dim);
	embeddings["<pad>"] = zeros(Float32, embedding_dim);

	# get embeddings only for tokens present in vocab (if Glove does not have embedding for token use randn)
	vocab_embeddings = map(token -> token in keys(embeddings) ? embeddings[token] : randn(Float32, embedding_dim), vocab)

	embedding_matrix = reduce(hcat, vocab_embeddings);

	@info "Encoding..."
    
	padding_idx = findfirst(x->x=="<pad>", vocab)

	

	# adjust length to MAX_LEN (pad or cut) and encode (replace tokens with their indices)
	encoded_train = ones(Int64,  length(tokenized_train), max_len) * padding_idx
	for i in 1:length(tokenized_train)
	    review = tokenized_train[i]
	    if max_len < length(review)
	        encoded_train[i,:] .= map(x->findfirst(y->y==x, vocab), review[1:max_len])
	    else
	        encoded_train[i,1:length(review)] .= map(x->findfirst(y->y==x, vocab), review)
	    end
	end

	encoded_test = ones(Int64,  length(tokenized_test), max_len) * padding_idx
	for i in 1:length(tokenized_test)
	    review = tokenized_test[i]
	    if max_len < length(review)
	        encoded_test[i,:] .= map(x->findfirst(y->y==x, vocab), review[1:max_len])
	    else
	        encoded_test[i,1:length(review)] .= map(x->findfirst(y->y==x, vocab), review)
	    end
	end

	X_train = encoded_train'
	X_test = encoded_test'
	y_train = reshape(labels_train, 1, :)
	y_test = reshape(labels_test, 1, :)

	return (;X_train, y_train, X_test, y_test, embeddings=embedding_matrix, vocab)
end
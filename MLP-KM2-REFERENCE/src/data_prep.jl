using JLD2
using TextAnalysis, Languages
using Random

function prepare_dataset(n_samples=10_000, split_ratio=0.8)
	@info "Loading raw dataset..."
	reviews = load("data/imdb_dataset.jld2", "reviews")[1:n_samples]
	labels = load("data/imdb_dataset.jld2", "labels")[1:n_samples]

	crps = Corpus(StringDocument.(reviews))
	languages!(crps, Languages.English())

	@info "Data preparation..."
	# get rid of anything that (probably) does not carry valuable information
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

	# get rid of words that occured less than min_freq
	min_freq = 5
	frequent_words = keys(filter(p->p[2] >= min_freq, lex))
	    
	reviews = map(x-> text(x), crps)

	for i in 1:length(reviews)
	    reviews[i] = join(filter(x-> x in frequent_words, split(reviews[i]))," ")
	end

	@info "Creating TF-IDF matrices..."
	# split dataset to train and test
	split_index = Int(floor(split_ratio * n_samples))

	reviews_train = reviews[1:split_index]
	labels_train = labels[1:split_index]

	reviews_test = reviews[split_index+1:n_samples]
	labels_test = labels[split_index+1:end]

	# prepare TF-IDF matrices
	train_crps = Corpus(StringDocument.(reviews_train))
	update_lexicon!(train_crps)
	test_crps = Corpus(StringDocument.(reviews_test))
	update_lexicon!(test_crps)

	dtm_train = DocumentTermMatrix(train_crps)
	dtm_test = DocumentTermMatrix(test_crps, dtm_train.terms)

	tfidf_train = Matrix{Float32}(tf_idf(dtm_train))
	tfidf_test = Matrix{Float32}(tf_idf(dtm_test))

	X_train = tfidf_train'
	X_test = tfidf_test'
	y_train = reshape(labels_train, 1, :)
	y_test = reshape(labels_test, 1, :)

	return (;X_train, y_train, X_test, y_test)
end
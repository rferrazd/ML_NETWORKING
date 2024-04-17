fn main() {
    println!("Hello, world!");

    let model_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoModelResources::GPT_NEO_2_7B,
    ));
    let config_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoVocabResources::GPT_NEO_2_7B,
    ));

    let merges_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoVocabResources::GPT_NEO_2_7B,
    ));

    // Text gerenration configuration structure
    let generate_config = TextGenerationConfig {
        model_type: ModelType::GPTNeo,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        num_beams: 5,
        no_repeat_ngram_size: 2,
        max_length: 100,
        ..Default::default() // set rest of fields to default
    };

    let model = TextGenerationModel::new(generate_config).unwrap();
    // add error handling capability

    // infinite loop: read, eval, print
    loop {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        // first token = prefix
        // remaining tokens = generation input
        // input will come after the "/" character
        let split = line.split('/').collect::<Vec<&str>>();
        let slc = split.as_slice();
        let output = model.generate(&slc[1..], Some(slc[0]));
        for sentence in output {
            println!("{}", sentence);
        }
    }

}   


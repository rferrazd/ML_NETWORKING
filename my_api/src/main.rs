// use Axum framework to make an API
//
// define route structure:
//      /       http-GET
//      /classes
//      /classes http-GET
//      /classes http-POST

// IMPORTANT NOTE:
// Given that neural-network project contains a Cargo.toml file and a src folder bc it's a Rust project. 
// To execute the neural-network program from the API code, need to build the Rust project first.
// SO.. navigate to the neural-network directory in terminal.
// Run: cargo build --release to compile the project in release mode. 
// This will generate an executable binary in the target/release directory.
////////////////////

// Import libs
use std::io;
use axum::{
    routing::{get,post},
    http::StatusCode,
    Json, Router,
};

use serde::{Deserialize, Serialize}; // these have to be added to Cargo.toml

// use Command module to execute the external Rust program (neural network) as a subprocess
use std::process::Command;



#[tokio::main]
async fn main() {
    // initialize tracing
    println!("Hello, world!");
    println!("Starting server...");
    tracing_subscriber::fmt::init(); // keep track of studd

    // build our application with a route
    let app = Router::new()
        // `GET /` goes to `root`
        // give the (endpoints = routes)
        // must have a root route. 
        .route("/", get(root)) // passing the functions that will be used as action handler. Wrap those functions around HTTPS action
        // `POST /users` goes to `create_user`
        
        //POST: This method is used to submit data to be processed to a specified resource. 
        .route("/neuralnetwork", post(run_neuralnetwork))
        .route("/neuralnetwork", get(get_neuralnetwork));
    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    // pass tcp lister to axum framework
    axum::serve(listener, app).await.unwrap();
}

// basic handler that responds with a static string
async fn root() -> &'static str {
    "Hello, World!" // last statemet = return value
}

/////////////////////////////////////////////////
//}

// Handler for GET /neuralnetwork endpoint
async fn get_neuralnetwork() -> Result<String, StatusCode> {
    println!("get_neuralnetwork function");
    // Call the run_neuralnetwork function
    let result = match run_neuralnetwork().await {
        Ok(result) => {
            println!("Matched"); // Print "Matched" if there is no error
            Ok(result)
        },
        Err(err) => {
            println!("Error occurred: {}", err);// Print "Error occurred" if there is an error
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        },
    };
    result
}


async fn run_neuralnetwork()-> Result<String, StatusCode> {
    println!("Inside run_neuralnetwork");
    // Execute your existing Rust program using Command
    // /Users/robertagarcia/Desktop/FINAL-PROJECT/neural-network/neural-network
    let output = Command::new("../neural-network/target/release/crabnn")
        .output()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?; // Return error if execution fails
    println!("This is the output");
    println!("This is the output: {}", String::from_utf8_lossy(&output.stdout)); 
    // Check if execution was successful
    if output.status.success() {
        // Convert output to a string and return it
        println!("Output of neural network is OK");
        let result = String::from_utf8(output.stdout).unwrap();
        Ok(result)

    } else {
        println!("Error in neural network");
        Err(StatusCode::INTERNAL_SERVER_ERROR)
    }
}


/*
async fn run_neuralnetwork() -> Result<String, StatusCode> {
    println!("Inside run_neuralnetwork");

    // Read user inputs from the terminal
    println!("Enter the first input:");
    let mut input1 = String::new();
    io::stdin().read_line(&mut input1).expect("Failed to read input");

    println!("Enter the second input:");
    let mut input2 = String::new();
    io::stdin().read_line(&mut input2).expect("Failed to read input");

    // Execute your existing Rust program using Command
    let output = Command::new("../neural-network/target/release/crabnn")
        .arg(input1.trim()) // Pass the first input as an argument
        .arg(input2.trim()) // Pass the second input as an argument
        .output()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Check if execution was successful
    if output.status.success() {
        // Convert output to a string and return it
        println!("Output of neural network is OK");
        let result = String::from_utf8(output.stdout).unwrap();
        Ok(result)
    } else {
        println!("Error in neural network");
        Err(StatusCode::INTERNAL_SERVER_ERROR)
    }
}
*/


#[derive(Deserialize)] // change this to whatever endpoint you have
struct CreateNN {
    name: String,
}

#[derive(Serialize)]
struct NN {
    crn: u64,
    name: String,
    // add more parameters. ex: description
}


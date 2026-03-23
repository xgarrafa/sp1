use sp1_sdk::{ProverClient, SP1Stdin};

fn main() {
    // 1. Setup the Prover (The Forge)
    let client = ProverClient::new();
    let mut stdin = SP1Stdin::new();

    // 2. Feed the Secret Inputs (Simulating BlockBasis Vault)
    stdin.write(&15000u64); // Your Balance
    stdin.write(&4500u64);  // Your Tax Liability

    println!("STARK/L: Initiating Proof of Solvency...");
    
    // 3. The Execution (This creates the 'ELF' and proves it)
    let (pk, vk) = client.setup(include_bytes!("../../program/elf/riscv32im-succinct-zkvm-elf"));
    let proof = client.prove(&pk, stdin).run().expect("Proving failed");

    // 4. The Result (The Signal for your Bento Grid)
    println!("STARK/L: Receipt Generated!");
    println!("Public Values (Proof): {:?}", proof.public_values);
}
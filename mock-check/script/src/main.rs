use sp1_sdk::{ProverClient, SP1Stdin};

fn main() {
    let client = ProverClient::new();
    let mut stdin = SP1Stdin::new();

    // TEST CASE: Spending $6,000 on a $5,000 budget
    let actual_spend: u32 = 6000; 
    let freelancer_id: u32 = 888; 
    let budget_limit: u32 = 5000; 
    
    stdin.write(&actual_spend);
    stdin.write(&freelancer_id);
    stdin.write(&budget_limit);

    println!("STARK/L: Generating Multi-Persona Tax & Guardrail Proof...");

    let (pk, vk) = client.setup(include_bytes!("../../program/elf/riscv32im-succinct-zkvm-elf"));
    let mut proof = client.prove(&pk, stdin).run().expect("Proving failed");

    // Read back the committed values in EXACT order
    let is_under_budget = proof.public_values.read::<bool>();
    let limit = proof.public_values.read::<u32>();
    let tax_impact = proof.public_values.read::<u32>();
    let id = proof.public_values.read::<u32>();

    println!("--- STARK/L: Receipt Generated ---");
    println!("Freelancer ID: #{}", id);
    println!("Is Under Budget: {}", is_under_budget);
    println!("Budget Limit: ${}", limit);
    println!("Suggested Tax Reserve: ${}", tax_impact);
    
    client.verify(&proof, &vk).expect("Verification failed");
    println!("STARK/L: Proof Mathematically Verified!");
}
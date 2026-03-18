#![no_main]
sp1_zkvm::entrypoint!(main);

use sp1_zkvm::syscalls::Poseidon2State;

pub fn main() {
    let mut state = Poseidon2State::default();

    // Hash ~10 MB: 436906 blocks of 24 bytes each = 10,485,744 bytes.
    for i in 0u32..436906 {
        let mut block = [0u8; 24];
        let bytes = i.to_le_bytes();
        block[..4].copy_from_slice(&bytes);
        state.absorb_byte_block(&block);
    }

    let output = state.output();
    println!("poseidon2 hash: {:?}", output);
}

const WIDTH: usize = 16;
const RATE: usize = 8;
const BYTE_BLOCK_SIZE: usize = 24;

use std::process::Output;

use elliptic_curve::ff::derive::bitvec::field;

use crate::syscall_poseidon2;

#[repr(C)]
#[repr(align(8))]
pub struct Poseidon2State([u32; WIDTH]);

impl Poseidon2State {
    #[inline]
    pub fn permute(&mut self) {
        unsafe {
            syscall_poseidon2(self);
        }
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u32 {
        self.0.as_mut_ptr()
    }

    fn absorb_field_block(&mut self, block: &[u32; RATE]) {
        self.0[0..RATE].copy_from_slice(block);
        self.permute();
    }

    pub fn absorb_block(&mut self, block: &[u8; BYTE_BLOCK_SIZE]) {
        let mut field_block = [0u32; RATE];
        // Accumulate every 24 bytes to a field element.
        for (i, element) in field_block.iter_mut().enumerate() {
            let start_idx = 3 * i;
            *element += block[start_idx] as u32;
            *element += (block[start_idx + 1] as u32) << 8;
            *element += (block[start_idx + 1] as u32) << 16;
        }
        self.absorb_field_block(&field_block);
    }

    pub fn output(self) -> [u32; RATE] {
        let mut output = [0; RATE];
        output.copy_from_slice(&self.0[0..RATE]);
        output
    }
}

struct DuplexSponge {
    state: Poseidon2State,
    absorbing_index: usize,
    squeezing_index: usize,
}

impl DuplexSponge {
    fn new() -> Self {
        let state = Poseidon2State([0; WIDTH]);

        Self { state, absorbing_index: 0, squeezing_index: RATE }
    }

    #[inline]
    fn absorb(&mut self, element: u32) {
        self.absorbing_index += 1;
        if self.absorbing_index == RATE {
            unsafe {
                syscall_poseidon2(&mut self.state);
            }
            self.absorbing_index = 0;
        }
        self.state.0[self.absorbing_index] = element;
    }

    #[inline]
    fn squeeze(&mut self) -> u32 {
        self.absorbing_index = 0;
        if self.squeezing_index == RATE {
            unsafe {
                syscall_poseidon2(&mut self.state);
            }
            self.squeezing_index = 0;
        }

        0
    }
}

const WIDTH: usize = 16;
const RATE: usize = 8;
const BYTE_BLOCK_SIZE: usize = 24;

use crate::syscall_poseidon2;

#[repr(C)]
#[repr(align(8))]
pub struct Poseidon2State([u32; WIDTH]);

impl Default for Poseidon2State {
    fn default() -> Self {
        Self([0; WIDTH])
    }
}

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

    /// Absorb a [`RATE`] size block of field elements
    ///
    /// # Safety
    /// This function assumes that the elements are within the `SP1Field` range. Breaking this
    /// constraint will lead to prover panic.
    unsafe fn absorb_field_block_unchecked(&mut self, block: &[u32; RATE]) {
        self.0[0..RATE].copy_from_slice(block);
        self.permute();
    }

    pub fn absorb_byte_block(&mut self, block: &[u8; BYTE_BLOCK_SIZE]) {
        let mut field_block = [0u32; RATE];
        // Accumulate every 24 bytes to a field element.
        for (i, element) in field_block.iter_mut().enumerate() {
            let start_idx = 3 * i;
            *element += block[start_idx] as u32;
            *element += (block[start_idx + 1] as u32) << 8;
            *element += (block[start_idx + 2] as u32) << 16;
        }
        unsafe {
            self.absorb_field_block_unchecked(&field_block);
        }
    }

    pub fn output(self) -> [u32; RATE] {
        let mut output = [0; RATE];
        output.copy_from_slice(&self.0[0..RATE]);
        output
    }
}

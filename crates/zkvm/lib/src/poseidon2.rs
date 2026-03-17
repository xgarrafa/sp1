const WIDTH: usize = 16;
const RATE: usize = 8;

#[repr(align(8))]
pub struct Poseidon2State([u32; 16]);

impl Poseidon2State {
    pub fn as_ptr(&self) -> *const u32 {
        self.0.as_ptr()
    }
}

struct DuplexSponge {
    state: Poseidon2State,
    absorbing_index: u64,
    squeezing_index: u64,
}

impl DuplexSponge {
    pub fn absorb(&mut self, element: u32) {}
}

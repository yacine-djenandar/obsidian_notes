
in the step based gradient compression, we do the following steps
1. We sort the gradients in ascending order
2. We generate a step value
3. We create steps approach based on bits to determine the value of the next gradients based on the next 1 in the bit set 
4. **Example**: the initial value is 0.5 and the step value is 0.1 => if we have a set of bits like the following: 1 0 0 1 0 1, we will then generate the gradients: 0.5, 2, 3
5. We will however need to send the indices with the bit array of steps to determine which gradient replaces which values.
6. As an intuition, We can use the step as the
## $$step=\frac{Max\ -\ Min}{Number\ of\ Gradients}$$
7. Example: for 1000 gradients: 
### $$\large{Number\ of\ bits=1000*32=32000\ bits}$$
	in our approach
### $$Number\ of\ bits=32+1000*8+bits\ representing\ steps$$
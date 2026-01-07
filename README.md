This folder contains codes for working with data collected from Keithley 2450 SourceMeter. 

The main use of these codes is for the data analysis of I-V characteristics of diodes at different temperatures. This experiment aims to quantify the temperature dependence of breakdown in 2.7 V and 9.1 V diodes and further investigate the behaviour of Zener parameter ($\beta$) and effective band gap at different temperatures. 

Reverse-bias I–V curves were measured for two diodes breakdown from liquid-nitrogen temperature to room temperature using four-wire sensing. Breakdown was defined operationally by the threshold voltage $V_{th}$ at which the reverse-current magnitude satisfies $|I| = 2.0\;\mathrm{mA}$, extracted using both fixed-current interpolation and a piecewise-linear (ReLU) knee fit to assess method dependence. 

The 2.7 V diode showed tunnelling-dominated behaviour: the fitted tunnelling scale decreased approximately linearly with temperature, ${\frac{d\beta Z}{dT} = (-8.528 \pm 0.300) \times 10^{-3}\;\mathrm{V\;K^{-1}}}$, implying a decreasing effective band-gap proxy with $\frac{dEg}{dT} =(-6.016 \pm 0.021) \times 10^{-4}\;\mathrm{eV\;K^{-1}}$. 

The 9.1 V diode exhibited the opposite $V_{th}(T)$ signature consistent with avalanche breakdown, although the extracted avalanche scale was less well constrained due to increased scatter and fit sensitivity.
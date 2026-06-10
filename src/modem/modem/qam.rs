use crate::modem::modem::*;

#[derive(Debug, Clone)]
pub(super) struct Qam {
    m_i: usize,       // bits per symbol, in-phase
    m_q: usize,       // bits per symbol, quadrature
    m_i_dim: usize,   // in-phase dimension, M_i=2^{m_i}
    m_q_dim: usize,   // quadrature dimension, M_q=2^{m_q}
    alpha: f32,       // scaling factor to ensure unity energy
}

impl Modem {
    pub(super) fn new_qam(bits_per_symbol: usize) -> Result<Self> {
        if bits_per_symbol < 1 {
            return Err(Error::Config("modem must have at least 2 bits/symbol".into()));
        }

        let (alpha, scheme) = match bits_per_symbol {
            2 => (1.0 / 2.0_f32.sqrt(), ModulationScheme::Qam4),
            3 => (1.0 / 6.0_f32.sqrt(), ModulationScheme::Qam8),
            4 => (1.0 / 10.0_f32.sqrt(), ModulationScheme::Qam16),
            5 => (1.0 / 26.0_f32.sqrt(), ModulationScheme::Qam32),
            6 => (1.0 / 42.0_f32.sqrt(), ModulationScheme::Qam64),
            7 => (1.0 / 106.0_f32.sqrt(), ModulationScheme::Qam128),
            8 => (1.0 / 170.0_f32.sqrt(), ModulationScheme::Qam256),
            _ => return Err(Error::Config("cannot support QAM with m > 8".into())),
        };

        let mut modem = Modem::_new(bits_per_symbol, scheme)?;

        let (m_i, m_q) = if bits_per_symbol % 2 == 1 {
            // rectangular qam
            ((bits_per_symbol + 1) >> 1, (bits_per_symbol - 1) >> 1)
        } else {
            // square qam
            (bits_per_symbol >> 1, bits_per_symbol >> 1)
        };

        let m_i_dim = 1 << m_i;
        let m_q_dim = 1 << m_q;

        assert_eq!(m_i + m_q, bits_per_symbol);
        assert_eq!(m_i_dim * m_q_dim, modem.constellation_size);

        let data = Qam {
            m_i,
            m_q,
            m_i_dim,
            m_q_dim,
            alpha,
        };

        modem.reference = Some([0.0; MAX_MOD_BITS_PER_SYMBOL]);
        let reference = modem.reference.as_mut().unwrap();

        for k in 0..modem.bits_per_symbol {
            reference[k] = (1 << k) as f32 * data.alpha;
        }

        modem.data = Some(ModemData::Qam(data));

        modem.symbol_map = Some(vec![Complex32::new(0.0, 0.0); modem.constellation_size]);
        modem.init_map()?;

        if modem.bits_per_symbol == 3 {
            modem.init_demod_soft_tab(3)?;
        } else if modem.bits_per_symbol >= 4 {
            modem.init_demod_soft_tab(4)?;
        }

        modem.reset();
        Ok(modem)
    }

    pub(super) fn modulate_qam(&mut self, symbol_in: u32) -> Result<Complex32> {
        if let ModemData::Qam(qam) = self.data.as_ref().unwrap() {
            let s_i = symbol_in >> qam.m_q;
            let s_q = symbol_in & ((1 << qam.m_q) - 1);

            // 'encode' symbols (actually gray decoding)
            let s_i = gray_decode(s_i);
            let s_q = gray_decode(s_q);

            // compute output sample
            let y = Complex32::new(
                (2 * s_i as i32 - qam.m_i_dim as i32 + 1) as f32 * qam.alpha,
                (2 * s_q as i32 - qam.m_q_dim as i32 + 1) as f32 * qam.alpha
            );
            Ok(y)
        } else {
            Err(Error::Internal("modem data is not of type Qam".into()))
        }
    }

    pub(super) fn demodulate_qam(&mut self, symbol_in: Complex32) -> Result<u32> {
        let (m_i, m_q) = match self.data.as_ref().unwrap() {
            ModemData::Qam(qam) => (qam.m_i, qam.m_q),
            _ => return Err(Error::Internal("modem data is not of type Qam".into())),
        };

        // demodulate in-phase component on linearly-spaced array
        let (s_i, res_i) = self.demodulate_linear_array_ref(symbol_in.re, m_i)?;

        // demodulate quadrature component on linearly-spaced array
        let (s_q, res_q) = self.demodulate_linear_array_ref(symbol_in.im, m_q)?;

        // 'decode' output symbol (actually gray encoding)
        let s_i = gray_encode(s_i);
        let s_q = gray_encode(s_q);
        let symbol_out = (s_i << m_q) + s_q;

        // re-modulate symbol (subtract residual) and store state
        self.x_hat = symbol_in - Complex32::new(res_i, res_q);
        self.r = symbol_in;
        Ok(symbol_out)
    }
}
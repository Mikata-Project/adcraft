use ndarray::prelude::*;
use num::clamp;
use numpy::{IntoPyArray, PyArray1, PyArrayDyn};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use pyo3::types::{PyDict, PyList};
use pyo3::{PyAny, PyObject, ToPyObject};
use rand::rngs::ThreadRng;
use rand::thread_rng;
use rand_distr::{Binomial, Distribution, Normal, StandardNormal};
use std::collections::HashMap;
use std::iter;

// The name of the module must be the same as the rust package name
#[pymodule]
fn rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn cost_mut<'py>(x_vec: &PyArrayDyn<f64>) {
        let mut x = unsafe { x_vec.as_array_mut() };
        // noise increasing with bid, and goes to 0 as bid goes to 0
        x.iter_mut().for_each(|p| {
            *p = clamp(
                (&p.sqrt() / 4.0 + *p / 2.0)
                    + Normal::new(0.0, 1e-10 + &p.sqrt() / 6.0)
                        .unwrap()
                        .sample(&mut thread_rng()),
                0.0,
                *p,
            )
        });
    }

    #[pyfn(m)]
    fn cost_trans<'py>(py: Python<'py>, x_vec: &PyArrayDyn<f64>) -> &'py PyArray1<f64> {
        let x = unsafe { x_vec.as_array() };
        // noise increasing with bid, and goes to 0 as bid goes to 0
        let result: Vec<f64> = x
            .iter()
            .map(|p| {
                clamp(
                    (p.sqrt() / 4.0 + p / 2.0)
                        + <StandardNormal as Distribution<f64>>::sample::<ThreadRng>(
                            &StandardNormal,
                            &mut thread_rng(),
                        ) * (1e-10 + p.sqrt() / 6.0),
                    0.0,
                    *p,
                )
            })
            .collect();
        result.into_pyarray(py)
    }

    #[pyfn(m)]
    fn cost_create<'py>(py: Python<'py>, x: f64, n: usize) -> &'py PyArray1<f64> {
        let mut result_vec = Array::from_elem((n,), 4.4);
        let x_sqrt = x.sqrt();
        // noise increasing with bid, and goes to 0 as bid goes to 0
        let normal = Normal::new(0.0, 1e-10 + &x_sqrt / 6.0).unwrap();
        result_vec.iter_mut().for_each(|p| {
            *p = clamp(
                (&x_sqrt / 4.0 + *p / 2.0) + normal.sample(&mut thread_rng()),
                0.0,
                *p,
            )
        });
        result_vec.into_pyarray(py)
    }

    #[pyfn(m)]
    fn binomial_impressions<'py>(n: u64, p: f64) -> u64 {
        // Original python function
        // self.rng.binomial(num_auctions, self.impression_rate(bid))

        let bin = Binomial::new(n, p).unwrap();
        bin.sample(&mut rand::thread_rng())
    }

    #[pyfn(m)]
    fn sigmoid<'py>(x: f64, s: f64, t: f64) -> f64 {
        // Original python function
        // 1.0 / (1.0 + np.exp(-s * (x - t)))
        rust_fn::sigmoid_rust(x, s, t)
    }

    #[pyfn(m)]
    fn probify_float<'py>(x: f64, y: f64, z: f64) -> f64 {
        // Original python function
        // np.clip(x, 0.0, 1.0)
        rust_fn::probify_rust(x, y, z)
    }

    #[pyfn(m)]
    fn threshold_sigmoid<'py>(p: f64, params: &PyDict) -> f64 {
        // params extraction
        let impression_thresh = rust_fn::get_value_with_default(params, "impression_thresh", 0.0);
        let impression_bid_threshold =
            rust_fn::get_value_with_default(params, "impression_bid_intercept", 0.1); // 50% impressions at this value
        let impression_slope = rust_fn::get_value_with_default(params, "impression_slope", 3.0); // tangent slope at bid threshold

        let halver = 2.0 + 1e-10;
        let thresh = rust_fn::probify_rust(halver * impression_thresh, 0.0, 1.0) / halver;
        let r = rust_fn::sigmoid_rust(p, impression_slope, impression_bid_threshold);

        rust_fn::probify_rust((1.0 + 2.0 * thresh) * r - thresh, 0.0, 1.0)
    }

    #[pyfn(m)]
    fn sum_array<'py>(x_vec: &PyArrayDyn<f64>) -> f64 {
        let x = unsafe { x_vec.as_array() };
        x.sum()
    }

    #[pyfn(m)]
    fn sum_list<'py>(x_vec: Vec<f64>) -> f64 {
        x_vec.iter().sum()
    }

    #[pyfn(m)]
    fn sum_list_bool<'py>(x_vec: Vec<bool>) -> usize {
        x_vec.into_iter().filter(|b| *b).count()
    }

    #[pyfn(m)]
    fn sum_array_bool<'py>(x_vec: &PyArrayDyn<bool>) -> usize {
        let x = unsafe { x_vec.as_array() };
        x.iter().filter(|b| **b).count()
    }

    #[pyfn(m)]
    fn array_to_zeros<'py>(py: Python<'py>, x_vec: &PyArrayDyn<f64>) -> &'py PyArray1<f64> {
        let x = unsafe { x_vec.as_array() };
        let result: Vec<f64> = x.iter().map(|_| 0.0).collect();
        result.into_pyarray(py)
    }

    #[pyfn(m)]
    fn list_to_zeros<'py>(py: Python<'py>, x_vec: Vec<f64>) -> &'py PyArray1<f64> {
        let result: Vec<f64> = x_vec.into_iter().map(|_| 0.0).collect();
        result.into_pyarray(py)
    }

    #[pyfn(m)]
    fn process_bidding_outcomes<'py>(
        py: Python<'py>,
        bidding_outcomes: &PyList,
        cumulative_profit: f64,
        loss_threshold: f64,
        current_day: i32,
        max_days: i32,
    ) -> PyResult<(HashMap<&'py str, PyObject>, f64, bool, i32, f64, bool)> {
        let mut impressions_list = Vec::new();
        let mut buyside_clicks_list = Vec::new();
        let mut cost_list: Vec<f64> = Vec::new();
        let mut sellside_conversions_list = Vec::new();
        let mut revenue_list = Vec::new();
        let mut profits_list = Vec::new();

        for kw in bidding_outcomes {
            let kw: &PyDict = kw.extract()?;
            impressions_list.push(kw.get_item("impressions").unwrap().extract::<i32>()?);
            buyside_clicks_list.push(kw.get_item("buyside_clicks").unwrap().extract::<i32>()?);
            cost_list.push(rust_fn::sum_list(
                kw.get_item("costs").unwrap().extract::<Vec<f64>>()?,
            ));
            sellside_conversions_list.push(
                kw.get_item("sellside_conversions")
                    .unwrap()
                    .extract::<i32>()?,
            );
            revenue_list.push(rust_fn::sum_list(
                kw.get_item("revenues").unwrap().extract::<Vec<f64>>()?,
            ));
            profits_list.push(kw.get_item("profit").unwrap().extract::<f64>()?);
        }

        let profits = rust_fn::sum_list(profits_list);
        let cumulative_profit = cumulative_profit + profits;
        // lost too much to keep bidding
        let truncated = cumulative_profit < -loss_threshold;

        let current_day = current_day + 1;
        let terminated = current_day >= max_days;

        let reward = profits;

        let mut observations: HashMap<&str, PyObject> = HashMap::new();
        observations.insert("impressions", impressions_list.into_pyarray(py).into());
        observations.insert(
            "buyside_clicks",
            buyside_clicks_list.into_pyarray(py).into(),
        );
        observations.insert("cost", cost_list.into_pyarray(py).into());
        observations.insert(
            "sellside_conversions",
            sellside_conversions_list.into_pyarray(py).into(),
        );
        observations.insert("revenue", revenue_list.into_pyarray(py).into());
        observations.insert(
            "cumulative_profit",
            vec![cumulative_profit].into_pyarray(py).into(),
        );
        observations.insert("days_passed", vec![current_day].into_pyarray(py).into());

        Ok((
            observations,
            cumulative_profit,
            truncated,
            current_day,
            reward,
            terminated,
        ))
    }

    // Prototype. Still calls Python so no speedup is actually occuring here.
    #[pyfn(m)]
    fn uniform_get_auctions_per_timestep<'py>(
        py: Python<'py>,
        timesteps: i32,
        kws: &PyList,
    ) -> PyResult<PyObject> {
        let mut volumes = Vec::new();
        let mut volume_step = Vec::new();
        for kw in kws {
            let kw = kw.downcast::<PyAny>()?;
            let sample_volume = kw.call_method0("sample_volume")?;
            let volume = sample_volume.get_item(0)?.extract::<i32>()?;
            volumes.push(volume);
            volume_step.push(volume / timesteps);
        }
        let first_timestep: Vec<i32> = volumes
            .iter()
            .zip(volume_step.iter())
            .map(|(vol, v)| vol - (timesteps - 1) * v)
            .collect();
        let remaining_timesteps = iter::repeat(volume_step)
            .take((timesteps - 1) as usize)
            .collect::<Vec<_>>();
        let auctions_per_timestep = iter::once(first_timestep)
            .chain(remaining_timesteps.into_iter())
            .map(|timestep| timestep.into_pyarray(py).to_object(py))
            .collect::<Vec<_>>();
        Ok(PyList::new(py, auctions_per_timestep).to_object(py))
    }

    #[pyfn(m)]
    fn nonneg_int_normal_sampler<'py>(the_mean: f64, std: f64) -> u64 {
        rust_fn::nonneg_int_normal_sampler(the_mean, std)
    }

    #[pyfn(m)]
    fn repr_outcomes_py<'py>(outcomes: &PyList) -> PyResult<String> {
        let mut result = String::from("[");
        for outcome in outcomes.iter() {
            let outcome = outcome.downcast::<PyDict>()?;
            let bid: f64 = outcome.get_item("bid").unwrap().extract()?;
            let impressions: i32 = outcome.get_item("impressions").unwrap().extract()?;
            let impression_share: f64 = outcome.get_item("impression_share").unwrap().extract()?;
            let buyside_clicks: i32 = outcome.get_item("buyside_clicks").unwrap().extract()?;
            let costs: Vec<f64> = outcome.get_item("costs").unwrap().extract()?;
            let sellside_conversions: i32 = outcome
                .get_item("sellside_conversions")
                .unwrap()
                .extract()?;
            let revenues: Vec<f64> = outcome.get_item("revenues").unwrap().extract()?;
            let revenues_per_cost: Vec<f64> =
                outcome.get_item("revenues_per_cost").unwrap().extract()?;
            let profit: f64 = outcome.get_item("profit").unwrap().extract()?;

            result.push_str(&format!("{{'bid': {}, 'impressions': {}, 'impression_share': {}, 'buyside_clicks': {}, 'costs': {:?}, 'sellside_conversions': {}, 'revenues': {:?}, 'revenues_per_cost': {:?}, 'profit': {}}}, ", bid, impressions, impression_share, buyside_clicks, costs, sellside_conversions, revenues, revenues_per_cost, profit));
        }
        result.pop();
        result.pop();
        result.push(']');
        Ok(result)
    }

    Ok(())
}

// The rust side functions
// Put it in mod to separate it from the python bindings
// These are just some random operations
// you probably want to do something more meaningful.
mod rust_fn {
    use num::clamp;
    use pyo3::types::PyDict;
    use rand::thread_rng;
    use rand_distr::{Distribution, Normal};

    pub fn sigmoid_rust(x: f64, s: f64, t: f64) -> f64 {
        // Original python function
        // 1.0 / (1.0 + np.exp(-s * (x - t)))
        1.0 / (1.0 + (-s * (x - t)).exp())
    }

    pub fn probify_rust(x: f64, y: f64, z: f64) -> f64 {
        // Original python function
        // np.clip(x, 0.0, 1.0)
        clamp(x, y, z)
    }

    pub fn get_value_with_default(dict: &PyDict, key: &str, default: f64) -> f64 {
        let value: Option<f64> = Some(dict.get_item(key).unwrap().extract().unwrap());
        match value {
            Some(v) => v,
            None => default,
        }
    }

    pub fn sum_list(x_vec: Vec<f64>) -> f64 {
        x_vec.iter().sum()
    }

    pub fn nonneg_int_normal_sampler(the_mean: f64, std: f64) -> u64 {
        let normal = Normal::new(the_mean, std).unwrap();
        // TODO: Use seed from int rather than random for reproducible results.
        //   Code below:
        // from rand::StdRng;
        // let mut r = StdRng::seed_from_u64(222);
        let result_raw = normal.sample(&mut thread_rng());
        // Non-negative
        let result_pos = clamp(result_raw, 0.0, result_raw);
        // Make int
        result_pos.round() as u64
    }
}

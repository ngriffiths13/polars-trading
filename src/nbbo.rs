// #![allow(clippy::unused_unit)]
// use polars::prelude::*;
// use pyo3_polars::derive::polars_expr;

// struct BBO {
//     bid: Option<f64>,
//     ask: Option<f64>,
// }

// #[polars_expr(output_type=Float64)]
// fn nbbo(inputs: &[Series]) -> PolarsResult<Series> {
//     let mut bbo_map: HashMap<u32, BBO> = HashMap::new();
//     let bid: &Float64Chunked = inputs[0].f64()?;
//     let ask: &Float64Chunked = inputs[1].f64()?;
//     let publisher_id: &UInt32Chunked = inputs[2].u32()?;
//     let bbos = Vec::with_capacity(bid.len());
//     for (i, (bid, ask, publisher_id)) in bid.into_iter().zip(ask).zip(publisher_id).enumerate() {
//         let bbo = bbo_map.entry(publisher_id).or_insert(BBO { bid: None, ask: None });
//         if bbo.bid.is_none() || bid > bbo.bid.unwrap() {
//             bbo.bid = Some(bid);
//         }
//         if bbo.ask.is_none() || ask < bbo.ask.unwrap() {
//             bbo.ask = Some(ask);
//         }
//         let best_bid = bbo_map.values().filter_map(|bbo| bbo.bid).max();
//         let best_ask = bbo_map.values().filter_map(|bbo| bbo.ask).min();
//         bbos.push(BBO { bid: best_bid, ask: best_ask });
//     }
//     let out = best_bid.zip(best_ask).map(|(bid, ask)| bid - ask);
//     Ok(out.into_series())
// }

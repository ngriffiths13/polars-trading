#![allow(clippy::unused_unit)]
use polars::{lazy::dsl::as_struct, prelude::*};
use pyo3_polars::derive::polars_expr;

/// This function calculates dynamic tick bar groups.
/// It takes a slice of Series as input and returns a PolarsResult of a Series with UInt16 type.
/// The function iterates over the thresholds and assigns a group id to each tick based on the threshold.
/// If the row count exceeds the threshold, the group id is incremented and the row count is reset.
#[polars_expr(output_type=UInt16)]
pub fn dynamic_tick_bar_groups(inputs: &[Series]) -> PolarsResult<Series> {
    let thresholds = inputs[0].u16()?;
    let mut group_id: u16 = 0;
    let mut row_count: u16 = 0;
    let mut builder: PrimitiveChunkedBuilder<UInt16Type> =
        PrimitiveChunkedBuilder::new("group_id", thresholds.len());
    for threshold in thresholds.into_iter() {
        match threshold {
            Some(threshold) => {
                row_count += 1;
                builder.append_value(group_id);
                if row_count >= threshold {
                    group_id += 1;
                    row_count = 0;
                }
            }
            None => builder.append_null(),
        }
    }
    Ok(builder.finish().into_series())
}

/// Struct representing a single transaction.
struct Transaction {
    dt: i64,
    price: f64,
    size: u32,
}

/// Struct representing OHLCV data.
struct OHLCV {
    start_dt: i64,
    end_dt: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    vwap: f64,
    volume: u32,
    n_transactions: u32,
}

/// Struct representing a collection of transactions.
struct BarTransactions {
    transactions: Vec<Transaction>,
}

impl BarTransactions {
    /// Create a new instance of BarTransactions.
    fn new() -> Self {
        Self {
            transactions: Vec::new(),
        }
    }

    /// Add a new transaction to the collection.
    fn add_transaction(&mut self, price: f64, size: u32, dt: i64) {
        self.transactions.push(Transaction { price, size, dt });
    }

    /// Check if the collection is empty.
    fn is_empty(&self) -> bool {
        self.transactions.is_empty()
    }

    /// Clear all transactions from the collection.
    fn clear_transactions(&mut self) {
        self.transactions.clear();
    }

    /// Get the current volume of the transactions.
    fn get_current_volume(&self) -> u32 {
        self.transactions.iter().map(|t| t.size).sum()
    }

    /// Get the current dollar volume of the transactions.
    fn get_current_dollar_volume(&self) -> f64 {
        self.transactions
            .iter()
            .map(|t| t.price * t.size as f64)
            .sum()
    }

    /// Calculate the OHLCV data from the transactions.
    fn calculate_ohlcv(&self) -> OHLCV {
        let start_dt = self.transactions.first().unwrap().dt;
        let end_dt = self.transactions.last().unwrap().dt;
        let open = self.transactions.first().unwrap().price;
        let close = self.transactions.last().unwrap().price;
        let high = self
            .transactions
            .iter()
            .map(|t| t.price)
            .fold(f64::MIN, f64::max);
        let low = self
            .transactions
            .iter()
            .map(|t| t.price)
            .fold(f64::MAX, f64::min);
        let volume = self.transactions.iter().map(|t| t.size).sum::<u32>();
        let vwap = self
            .transactions
            .iter()
            .map(|t| t.price * t.size as f64)
            .sum::<f64>()
            / volume as f64;
        let n_transactions = self.transactions.len().try_into().unwrap();
        OHLCV {
            start_dt,
            end_dt,
            open,
            high,
            low,
            close,
            vwap,
            volume,
            n_transactions,
        }
    }
}

/// Enum to represent the threshold for calculating bars.
enum Threshold {
    Volume(u32), // Threshold based on volume
    Dollar(f64), // Threshold based on dollar value
}

/// Implementation of the Threshold enum.
impl Threshold {
    /// Create a Threshold enum from a u32 value.
    fn from_u32(threshold: Option<u32>) -> Option<Self> {
        threshold.map(Threshold::Volume)
    }

    /// Create a Threshold enum from a f64 value.
    fn from_f64(threshold: Option<f64>) -> Option<Self> {
        threshold.map(Threshold::Dollar)
    }
}

/// Function to calculate bars from trades.
/// It takes in datetimes, prices, sizes, and thresholds as input.
/// It returns a DataFrame containing the calculated bars.
fn calculate_bars_from_trades(
    datetimes: &[Option<i64>], // Datetimes of the trades
    prices: &[Option<f64>], // Prices of the trades
    sizes: &[Option<u32>], // Sizes of the trades
    threshold: &[Option<Threshold>], // Threshold for calculating the bars
) -> PolarsResult<DataFrame> {
    // TODO: Add dollar volume to OHLCV
    let mut bars: Vec<OHLCV> = Vec::new(); // Vector to store the calculated bars
    let mut start_dt: Vec<i64> = Vec::new(); // Vector to store the start datetimes of the bars
    let mut end_dt: Vec<i64> = Vec::new(); // Vector to store the end datetimes of the bars
    let mut opens: Vec<f64> = Vec::new(); // Vector to store the opening prices of the bars
    let mut highs: Vec<f64> = Vec::new(); // Vector to store the highest prices of the bars
    let mut lows: Vec<f64> = Vec::new(); // Vector to store the lowest prices of the bars
    let mut closes: Vec<f64> = Vec::new(); // Vector to store the closing prices of the bars
    let mut vwap: Vec<f64> = Vec::new(); // Vector to store the volume weighted average prices of the bars
    let mut volumes: Vec<u32> = Vec::new(); // Vector to store the volumes of the bars
    let mut n_transactions: Vec<u32> = Vec::new(); // Vector to store the number of transactions in the bars
    let mut bar_transactions = BarTransactions::new(); // BarTransactions instance to calculate the bars

    // CALCULATE BARS AND ADD TO SERIES THEN CREATE DF
    for (((dt, price), size), thresh) in datetimes
        .iter()
        .zip(prices.iter())
        .zip(sizes.iter())
        .zip(threshold.iter())
    {
        match (dt, price, size, thresh) {
            (Some(dt), Some(price), Some(mut size), Some(thresh)) => match thresh {
                Threshold::Volume(thresh) => {
                    if size >= thresh - bar_transactions.get_current_volume() {
                        let mut remaining_size = thresh - bar_transactions.get_current_volume();
                        while size >= remaining_size {
                            bar_transactions.add_transaction(*price, remaining_size, *dt);
                            let ohlcv = bar_transactions.calculate_ohlcv();
                            start_dt.push(ohlcv.start_dt);
                            end_dt.push(ohlcv.end_dt);
                            opens.push(ohlcv.open);
                            highs.push(ohlcv.high);
                            lows.push(ohlcv.low);
                            closes.push(ohlcv.close);
                            vwap.push(ohlcv.vwap);
                            volumes.push(ohlcv.volume);
                            n_transactions.push(ohlcv.n_transactions);
                            bars.push(ohlcv);
                            bar_transactions.clear_transactions();
                            size -= remaining_size;
                            remaining_size = *thresh;
                        }
                        if size > 0 {
                            bar_transactions.add_transaction(*price, size, *dt);
                        }
                    } else {
                        bar_transactions.add_transaction(*price, size, *dt);
                    }
                }
                Threshold::Dollar(thresh) => {
                    if price * size as f64
                        >= *thresh - bar_transactions.get_current_dollar_volume() as f64
                    {
                        let mut remaining = *thresh - bar_transactions.get_current_dollar_volume();
                        while price * size as f64 >= remaining {
                            bar_transactions.add_transaction(
                                *price,
                                (remaining / *price) as u32,
                                *dt,
                            );
                            let ohlcv = bar_transactions.calculate_ohlcv();
                            start_dt.push(ohlcv.start_dt);
                            end_dt.push(ohlcv.end_dt);
                            opens.push(ohlcv.open);
                            highs.push(ohlcv.high);
                            lows.push(ohlcv.low);
                            closes.push(ohlcv.close);
                            vwap.push(ohlcv.vwap);
                            volumes.push(ohlcv.volume);
                            n_transactions.push(ohlcv.n_transactions);
                            bars.push(ohlcv);
                            bar_transactions.clear_transactions();
                            size -= (remaining / *price) as u32;
                            remaining = *thresh;
                        }
                        if size > 0 {
                            bar_transactions.add_transaction(*price, size, *dt);
                        }
                    } else {
                        bar_transactions.add_transaction(*price, size, *dt);
                    }
                }
            },
            _ => {}
        }
        // create an array of len bars and fill with symbol
    }
    if !bar_transactions.is_empty() {
        let ohlcv = bar_transactions.calculate_ohlcv();
        start_dt.push(ohlcv.start_dt);
        end_dt.push(ohlcv.end_dt);
        opens.push(ohlcv.open);
        highs.push(ohlcv.high);
        lows.push(ohlcv.low);
        closes.push(ohlcv.close);
        vwap.push(ohlcv.vwap);
        volumes.push(ohlcv.volume);
        n_transactions.push(ohlcv.n_transactions);
        bars.push(ohlcv);
    }

    // Create a DataFrame from the calculated bars
    df!(
        "start_dt" => Series::from_vec("start_dt", start_dt),
        "end_dt" => Series::from_vec("end_dt", end_dt),
        "open" => Series::from_vec("open", opens),
        "high" => Series::from_vec("high", highs),
        "low" => Series::from_vec("low", lows),
        "close" => Series::from_vec("close", closes),
        "vwap" => Series::from_vec("vwap", vwap),
        "volume" => Series::from_vec("volume", volumes),
        "n_transactions" => Series::from_vec("n_transactions", n_transactions)
    )
}

/// Function to define the type of the OHLCV struct.
/// It takes in the input fields of the DataFrame.
/// It returns a Field representing the OHLCV struct.
fn ohlcv_struct_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "ohlcv",
        DataType::Struct(vec![
            Field::new("start_dt", DataType::Datetime(TimeUnit::Nanoseconds, None)),
            Field::new("end_dt", DataType::Datetime(TimeUnit::Nanoseconds, None)),
            Field::new("open", DataType::Float64),
            Field::new("high", DataType::Float64),
            Field::new("low", DataType::Float64),
            Field::new("close", DataType::Float64),
            Field::new("vwap", DataType::Float64),
            Field::new("volume", DataType::UInt32),
            Field::new("n_transactions", DataType::UInt32),
        ]),
    ))
}

/// Function to calculate volume bars.
/// It takes in a list of Series as input.
/// It returns a Series containing the calculated volume bars.
#[polars_expr(output_type_func=ohlcv_struct_type)] // FIXME
pub fn volume_bars(inputs: &[Series]) -> PolarsResult<Series> {
    let dts = inputs[0].datetime()?; // Datetimes of the trades
    let dt_type = dts.dtype(); // Type of the datetimes
    let dts = dts.to_vec(); // Convert the datetimes to a vector
    let prices = inputs[1].f64()?.to_vec(); // Prices of the trades
    let sizes = inputs[2].u32()?.to_vec(); // Sizes of the trades
    let threshold = inputs[3].u32()?.to_vec(); // Threshold for calculating the bars
    let threshold = threshold
        .iter()
        .map(|&x| Threshold::from_u32(x))
        .collect::<Vec<Option<Threshold>>>(); // Convert the threshold to a vector of Threshold enums

    // Calculate the bars from the trades
    let bars = calculate_bars_from_trades(
        dts.as_slice(),
        prices.as_slice(),
        sizes.as_slice(),
        threshold.as_slice(),
    )?;
    let s = bars
        .lazy()
        .with_columns(vec![
            col("start_dt").cast(dt_type.clone()), // Cast the start datetimes to the original type
            col("end_dt").cast(dt_type.clone()), // Cast the end datetimes to the original type
        ])
        .select([as_struct(vec![
            col("start_dt"),
            col("end_dt"),
            col("open"),
            col("high"),
            col("low"),
            col("close"),
            col("vwap"),
            col("volume"),
            col("n_transactions"),
        ])
        .alias("bar")])
        .collect()?
        .column("bar")?
        .clone(); // Select the OHLCV struct and cast it to the original type
    Ok(s) // Return the calculated bars
}


/// Function to calculate dollar bars.
/// It takes in a list of Series as input.
/// It returns a Series containing the calculated dollar bars.
#[polars_expr(output_type_func=ohlcv_struct_type)]
pub fn dollar_bars(inputs: &[Series]) -> PolarsResult<Series> {
    let dts = inputs[0].datetime()?; // Datetimes of the trades
    let dt_type = dts.dtype(); // Type of the datetimes
    let dts = dts.to_vec(); // Convert the datetimes to a vector
    let prices = inputs[1].f64()?.to_vec(); // Prices of the trades
    let sizes = inputs[2].u32()?.to_vec(); // Sizes of the trades
    let threshold = inputs[3].f64()?.to_vec(); // Threshold for calculating the bars
    let threshold = threshold
        .iter()
        .map(|&x| Threshold::from_f64(x))
        .collect::<Vec<Option<Threshold>>>(); // Convert the threshold to a vector of Threshold enums
    let bars = calculate_bars_from_trades(
        dts.as_slice(),
        prices.as_slice(),
        sizes.as_slice(),
        threshold.as_slice(),
    )?; // Calculate the bars from the trades
    let s = bars
        .lazy()
        .with_columns(vec![
            col("start_dt").cast(dt_type.clone()), // Cast the start datetimes to the original type
            col("end_dt").cast(dt_type.clone()), // Cast the end datetimes to the original type
        ])
        .select([as_struct(vec![
            col("start_dt"),
            col("end_dt"),
            col("open"),
            col("high"),
            col("low"),
            col("close"),
            col("vwap"),
            col("volume"),
            col("n_transactions"),
        ])
        .alias("bar")])
        .collect()?
        .column("bar")?
        .clone(); // Select the OHLCV struct and cast it to the original type
    Ok(s) // Return the calculated bars
}

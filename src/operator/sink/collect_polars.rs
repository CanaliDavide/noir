use std::fmt::Display;
use std::marker::PhantomData;

use polars::prelude::{AnyValue, DataFrame};
use polars::series::Series;

use crate::block::{BlockStructure, OperatorKind, OperatorStructure};
use crate::operator::sink::{Sink, StreamOutputRef};
use crate::operator::{ExchangeData, Operator, StreamElement};
use crate::scheduler::ExecutionMetadata;

#[derive(Debug)]
pub struct CollectPolars<'a, Out: ExchangeData, PreviousOperators, F>
where
    PreviousOperators: Operator<Out>,
    F: Fn(Out) -> AnyValue<'a> + Send,
{
    prev: PreviousOperators,
    result: Option<Vec<AnyValue<'a>>>,
    output: StreamOutputRef<DataFrame>,
    get_value: F,
    column_name: String,
    _o: PhantomData<Out>,
}

impl<'a, Out: ExchangeData, PreviousOperators, F> CollectPolars<'a, Out, PreviousOperators, F>
where
    PreviousOperators: Operator<Out>,
    F: Fn(Out) -> AnyValue<'a> + Send,
{
    pub(crate) fn new(
        prev: PreviousOperators,
        output: StreamOutputRef<DataFrame>,
        column_name: String,
        get_value: F,
        _o: PhantomData<Out>,
    ) -> Self {
        Self {
            prev,
            result: Some(Vec::new()),
            output,
            get_value,
            column_name,
            _o,
        }
    }
}

impl<'a, Out: ExchangeData, PreviousOperators, F> Display
    for CollectPolars<'a, Out, PreviousOperators, F>
where
    PreviousOperators: Operator<Out>,
    F: Fn(Out) -> AnyValue<'a> + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> CollectPolars", self.prev)
    }
}

impl<'a, Out: ExchangeData, PreviousOperators, F> Operator<()>
    for CollectPolars<'a, Out, PreviousOperators, F>
where
    PreviousOperators: Operator<Out>,
    F: Fn(Out) -> AnyValue<'a> + Send,
{
    fn setup(&mut self, metadata: &mut ExecutionMetadata) {
        self.prev.setup(metadata);
    }

    fn next(&mut self) -> StreamElement<()> {
        match self.prev.next() {
            StreamElement::Item(t) | StreamElement::Timestamped(t, _) => {
                // cloned CollectVecSink or already ended stream
                if let Some(result) = self.result.as_mut() {
                    result.push((self.get_value)(t));
                }
                StreamElement::Item(())
            }
            StreamElement::Watermark(w) => StreamElement::Watermark(w),
            StreamElement::Terminate => {
                if let Some(result) = self.result.take() {
                    *self.output.lock().unwrap() = Some(
                        DataFrame::new(vec![Series::from_any_values(
                            &self.column_name,
                            &result,
                            false,
                        )
                        .unwrap()])
                        .unwrap(),
                    )
                }
                StreamElement::Terminate
            }
            StreamElement::FlushBatch => StreamElement::FlushBatch,
            StreamElement::FlushAndRestart => StreamElement::FlushAndRestart,
        }
    }

    fn structure(&self) -> BlockStructure {
        let mut operator = OperatorStructure::new::<Out, _>("CollectVecSink");
        operator.kind = OperatorKind::Sink;
        self.prev.structure().add_operator(operator)
    }
}

impl<'a, Out: ExchangeData, PreviousOperators, F> Sink
    for CollectPolars<'a, Out, PreviousOperators, F>
where
    PreviousOperators: Operator<Out>,
    F: Fn(Out) -> AnyValue<'a> + Send,
{
}

impl<'a, Out: ExchangeData, PreviousOperators, F> Clone
    for CollectPolars<'a, Out, PreviousOperators, F>
where
    PreviousOperators: Operator<Out>,
    F: Fn(Out) -> AnyValue<'a> + Send,
{
    fn clone(&self) -> Self {
        panic!("CollectPolars cannot be cloned, max_parallelism should be 1");
    }
}

#[cfg(test)]
mod tests {

    use polars::prelude::{AnyValue, DataFrame, NamedFrom};
    use polars::series::Series;

    use crate::config::EnvironmentConfig;
    use crate::environment::StreamEnvironment;
    use crate::operator::source;

    #[test]
    fn collect_vec() {
        let mut env = StreamEnvironment::new(EnvironmentConfig::local(4));
        let source = source::IteratorSource::new(0..10i32);
        let res = env
            .stream(source)
            .collect_polars("test".to_string(), AnyValue::Int32);
        env.execute();
        let d = res.get().unwrap();
        assert_eq!(
            d,
            DataFrame::new(vec![Series::new("test", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]).unwrap()
        );
    }
}

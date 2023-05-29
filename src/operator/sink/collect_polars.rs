use std::{fmt::Display, marker::PhantomData};

use polars::{
    prelude::{DataFrame, NamedFromOwned},
    series::Series,
};
use serde::{Deserialize, Serialize};

use crate::{
    block::{BlockStructure, OperatorKind, OperatorStructure},
    operator::{Operator, StreamElement},
    ExecutionMetadata,
};

use super::{Sink, StreamOutputRef};

#[derive(Clone, Serialize, Deserialize, Debug)]
enum NoirType {
    Int32(i32),
    Float32(f32),
    Row(Row),
}

#[derive(Clone, Serialize, Deserialize, Debug)]
struct Row {
    columns: Vec<NoirType>,
}

#[derive(Debug)]
pub struct CollectPolars<T, PreviousOperators> {
    prev: PreviousOperators,
    result: Option<Vec<NoirType>>,
    output: StreamOutputRef<DataFrame>,
    column_name: String,
    _a: PhantomData<T>,
}

macro_rules! collect_polars_generic {
    ($type:ty, $noirType:ident ) => {
        impl<PreviousOperators> CollectPolars<$type, PreviousOperators>
        where
            PreviousOperators: Operator<$type>,
        {
            pub(crate) fn new(
                prev: PreviousOperators,
                output: StreamOutputRef<DataFrame>,
                column_name: String,
            ) -> Self {
                Self {
                    prev,
                    result: Some(Vec::new()),
                    output,
                    column_name,
                    _a: PhantomData,
                }
            }
        }

        impl<PreviousOperators> Operator<()> for CollectPolars<$type, PreviousOperators>
        where
            PreviousOperators: Operator<$type>,
        {
            fn setup(&mut self, metadata: &mut ExecutionMetadata) {
                self.prev.setup(metadata);
            }

            fn next(&mut self) -> StreamElement<()> {
                match self.prev.next() {
                    StreamElement::Item(t) | StreamElement::Timestamped(t, _) => {
                        // cloned CollectVecSink or already ended stream
                        if let Some(result) = self.result.as_mut() {
                            result.push(NoirType::$noirType(t));
                        }
                        StreamElement::Item(())
                    }
                    StreamElement::Watermark(w) => StreamElement::Watermark(w),
                    StreamElement::Terminate => {
                        if let Some(result) = self.result.take() {
                            let res: Vec<$type> = result
                                .iter()
                                .map(|v| match v {
                                    NoirType::$noirType(i) => *i,
                                    _ => panic!("Type mismatch!"),
                                })
                                .collect();

                            *self.output.lock().unwrap() = Some(
                                DataFrame::new(vec![Series::from_vec(&self.column_name, res)])
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
                let mut operator = OperatorStructure::new::<$type, _>("CollectVecSink");
                operator.kind = OperatorKind::Sink;
                self.prev.structure().add_operator(operator)
            }
        }

        impl<PreviousOperators> Display for CollectPolars<$type, PreviousOperators>
        where
            PreviousOperators: Operator<$type>,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{} -> CollectPolars", self.prev)
            }
        }

        impl<PreviousOperators> Clone for CollectPolars<$type, PreviousOperators>
        where
            PreviousOperators: Operator<$type>,
        {
            fn clone(&self) -> Self {
                panic!("CollectPolars cannot be cloned, max_parallelism should be 1");
            }
        }

        impl<PreviousOperators> Sink for CollectPolars<$type, PreviousOperators> where
            PreviousOperators: Operator<$type>
        {
        }
    };
}

collect_polars_generic!(i32, Int32);
collect_polars_generic!(f32, Float32);

#[cfg(test)]
mod tests {

    use polars::prelude::{DataFrame, NamedFrom};
    use polars::series::Series;

    use crate::config::EnvironmentConfig;
    use crate::environment::StreamEnvironment;
    use crate::operator::source;

    #[test]
    fn collect_vec() {
        let mut env = StreamEnvironment::new(EnvironmentConfig::local(4));
        let source = source::IteratorSource::new(0..10i32);
        let res = env.stream(source).collect_polars("test".to_string());
        env.execute();
        let d = res.get().unwrap();
        assert_eq!(
            d,
            DataFrame::new(vec![Series::new("test", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]).unwrap()
        );
    }
}

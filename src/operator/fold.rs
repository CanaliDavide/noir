use std::sync::Arc;

use crate::block::NextStrategy;
use crate::operator::{Data, EndBlock, Operator, StreamElement, Timestamp};
use crate::scheduler::ExecutionMetadata;
use crate::stream::Stream;

#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct Fold<Out: Data, NewOut: Data, PreviousOperators>
where
    PreviousOperators: Operator<Out>,
{
    prev: PreviousOperators,
    #[derivative(Debug = "ignore")]
    fold: Arc<dyn Fn(NewOut, Out) -> NewOut + Send + Sync>,
    init: NewOut,
    accumulator: Option<NewOut>,
    timestamp: Option<Timestamp>,
    max_watermark: Option<Timestamp>,
    received_end: bool,
}

impl<Out: Data, NewOut: Data, PreviousOperators> Operator<NewOut>
    for Fold<Out, NewOut, PreviousOperators>
where
    PreviousOperators: Operator<Out> + Send,
{
    fn setup(&mut self, metadata: ExecutionMetadata) {
        self.prev.setup(metadata);
    }

    fn next(&mut self) -> StreamElement<NewOut> {
        while !self.received_end {
            match self.prev.next() {
                StreamElement::End => self.received_end = true,
                StreamElement::Watermark(ts) => {
                    self.max_watermark = Some(self.max_watermark.unwrap_or(ts).max(ts))
                }
                StreamElement::Item(item) => {
                    self.accumulator = Some((self.fold)(
                        self.accumulator.take().unwrap_or_else(|| self.init.clone()),
                        item,
                    ));
                }
                StreamElement::Timestamped(item, ts) => {
                    self.timestamp = Some(self.timestamp.unwrap_or(ts).max(ts));
                    self.accumulator = Some((self.fold)(
                        self.accumulator.take().unwrap_or_else(|| self.init.clone()),
                        item,
                    ));
                }
                // this block wont sent anything until the stream ends
                StreamElement::FlushBatch => {}
            }
        }

        // If there is an accumulated value, return it
        if let Some(acc) = self.accumulator.take() {
            if let Some(ts) = self.timestamp.take() {
                return StreamElement::Timestamped(acc, ts);
            } else {
                return StreamElement::Item(acc);
            }
        }

        // If watermark were received, send one downstream
        if let Some(ts) = self.max_watermark.take() {
            return StreamElement::Watermark(ts);
        }

        StreamElement::End
    }

    fn to_string(&self) -> String {
        format!(
            "{} -> Fold<{} -> {}>",
            self.prev.to_string(),
            std::any::type_name::<Out>(),
            std::any::type_name::<NewOut>()
        )
    }
}

impl<Out: Data, OperatorChain> Stream<Out, OperatorChain>
where
    OperatorChain: Operator<Out> + Send + 'static,
{
    pub fn fold<NewOut: Data, Local, Global>(
        self,
        init: NewOut,
        local: Local,
        global: Global,
    ) -> Stream<NewOut, impl Operator<NewOut>>
    where
        Local: Fn(NewOut, Out) -> NewOut + Send + Sync + 'static,
        Global: Fn(NewOut, NewOut) -> NewOut + Send + Sync + 'static,
    {
        // Local fold
        let mut second_part = self
            .add_operator(|prev| Fold {
                prev,
                fold: Arc::new(local),
                init: init.clone(),
                accumulator: None,
                timestamp: None,
                max_watermark: None,
                received_end: false,
            })
            .add_block(EndBlock::new, NextStrategy::OnlyOne);

        // Global fold (which is done on only one node)
        second_part.block.scheduler_requirements.max_parallelism(1);
        second_part.add_operator(|prev| Fold {
            prev,
            fold: Arc::new(global),
            init,
            accumulator: None,
            timestamp: None,
            max_watermark: None,
            received_end: false,
        })
    }
}

#[cfg(test)]
mod tests {

    use crate::config::EnvironmentConfig;
    use crate::environment::StreamEnvironment;
    use crate::operator::source;

    #[test]
    fn fold_stream() {
        let mut env = StreamEnvironment::new(EnvironmentConfig::local(4));
        let source = source::StreamSource::new(0..10u8);
        let res = env
            .stream(source)
            .fold("".to_string(), |s, n| s + &n.to_string(), |s1, s2| s1 + &s2)
            .collect_vec();
        env.execute();
        let res = res.get().unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], "0123456789");
    }
}

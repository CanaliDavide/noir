use flume::{Receiver, bounded, Sender};

use crate::block::{BlockStructure, OperatorKind, OperatorStructure};
use crate::operator::source::Source;
use crate::operator::{Data, Operator, StreamElement};
use crate::scheduler::ExecutionMetadata;

/// Source that consumes an iterator and emits all its elements into the stream.
///
/// The iterator will be consumed **only from one replica**, therefore this source is not parallel.

#[derive(Derivative)]
#[derivative(Debug)]
pub struct ChannelSource<Out: Data> {
    #[derivative(Debug = "ignore")]
    rx: Receiver<Out>,
    terminated: bool,
}

impl<Out: Data> ChannelSource<Out> {
    /// Create a new source that reads the items from the iterator provided as input.
    ///
    /// **Note**: this source is **not parallel**, the iterator will be consumed only on a single
    /// replica, on all the others no item will be read from the iterator. If you want to achieve
    /// parallelism you need to add an operator that shuffles the data (e.g.
    /// [`Stream::shuffle`](crate::Stream::shuffle)).
    ///
    /// ## Example
    ///
    /// ```
    /// # use noir::{StreamEnvironment, EnvironmentConfig};
    /// # use noir::operator::source::ChannelSource;
    /// # let mut env = StreamEnvironment::new(EnvironmentConfig::local(1));
    /// let source = ChannelSource::new((0..5));
    /// let R = env.stream(source);
    /// ```
    pub fn new(channel_size: usize) -> (Sender<Out>, Self) {
        let (tx, rx) = bounded(channel_size);
        let s = Self {
            rx,
            terminated: false,
        };

        (tx, s)
    }
}
// TODO: remove Debug requirement
impl<Out: Data + core::fmt::Debug> Source<Out> for ChannelSource<Out> {
    fn get_max_parallelism(&self) -> Option<usize> {
        Some(1)
    }
}

impl<Out: Data + core::fmt::Debug> Operator<Out> for ChannelSource<Out> {
    fn setup(&mut self, _metadata: ExecutionMetadata) {}

    fn next(&mut self) -> StreamElement<Out> {
        if self.terminated {
            return StreamElement::Terminate;
        }
        // TODO: with adaptive batching this does not work since R never emits FlushBatch messages
        log::warn!("waiting for value on channel");
        match self.rx.recv() {
            Ok(t) => {
                log::warn!("received value, forwarding");
                StreamElement::Item(t)
            },
            Err(e) => {
                self.terminated = true;
                log::error!("Error in source stream: {e}"); // TODO change
                StreamElement::FlushAndRestart
            }
        }
    }

    fn to_string(&self) -> String {
        format!("StreamSource<{}>", std::any::type_name::<Out>())
    }

    fn structure(&self) -> BlockStructure {
        let mut operator = OperatorStructure::new::<Out, _>("ChannelSource");
        operator.kind = OperatorKind::Source;
        BlockStructure::default().add_operator(operator)
    }
}

impl<Out: Data> Clone for ChannelSource<Out> {
    fn clone(&self) -> Self {
        // Since this is a non-parallel source, we don't want the other replicas to emit any value
        panic!("ChannelSource cannot be cloned, max_parallelism should be 1");
    }
}
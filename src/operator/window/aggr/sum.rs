use std::ops::AddAssign;

use super::{super::*, Fold};
use crate::operator::{Data, DataKey, Operator};
use crate::stream::{KeyValue, KeyedStream, WindowedStream};

impl<Key, Out, WindowDescr, OperatorChain> WindowedStream<Key, Out, OperatorChain, Out, WindowDescr>
where
    WindowDescr: WindowBuilder,
    OperatorChain: Operator<KeyValue<Key, Out>> + 'static,
    Key: DataKey,
    Out: Data,
{
    pub fn sum<NewOut: Data + Default + AddAssign<Out>>(
        self,
    ) -> KeyedStream<Key, NewOut, impl Operator<KeyValue<Key, NewOut>>> {
        let acc = Fold::new(NewOut::default(), |sum, x| *sum += x);
        self.add_window_operator("WindowSum", acc)
    }
}

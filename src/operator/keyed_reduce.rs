use crate::operator::{Data, DataKey, Operator};
use crate::stream::{KeyValue, KeyedStream, Stream};
use std::sync::Arc;

impl<Out: Data, OperatorChain> Stream<Out, OperatorChain>
where
    OperatorChain: Operator<Out> + Send + 'static,
{
    pub fn group_by_reduce<Key: DataKey, Keyer, F>(
        self,
        keyer: Keyer,
        f: F,
    ) -> KeyedStream<Key, Out, impl Operator<KeyValue<Key, Out>>>
    where
        Keyer: Fn(&Out) -> Key + Send + Sync + 'static,
        F: Fn(Out, Out) -> Out + Send + Sync + 'static,
    {
        // FIXME: remove Arc if reduce function will be Clone
        let f = Arc::new(f);
        let f2 = f.clone();

        self.group_by_fold(
            keyer,
            None,
            move |acc, value| match acc {
                None => Some(value),
                Some(acc) => Some(f(acc, value)),
            },
            move |acc1, acc2| match acc1 {
                None => acc2,
                Some(acc1) => match acc2 {
                    None => Some(acc1),
                    Some(acc2) => Some(f2(acc1, acc2)),
                },
            },
        )
        .map(|(_, value)| value.unwrap())
    }
}

impl<Key: DataKey, Out: Data, OperatorChain> KeyedStream<Key, Out, OperatorChain>
where
    OperatorChain: Operator<KeyValue<Key, Out>> + Send + 'static,
{
    pub fn reduce<F>(self, f: F) -> KeyedStream<Key, Out, impl Operator<KeyValue<Key, Out>>>
    where
        F: Fn(Out, Out) -> Out + Send + Sync + 'static,
    {
        self.fold(None, move |acc, value| match acc {
            None => Some(value),
            Some(acc) => Some(f(acc, value)),
        })
        .map(|(_, value)| value.unwrap())
    }
}

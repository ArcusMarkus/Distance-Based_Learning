package learning.classifiers;

import core.Duple;
import learning.core.Classifier;
import learning.core.Histogram;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.function.ToDoubleBiFunction;

// KnnTest.test() should pass once this is finished.
public class Knn<V, L> implements Classifier<V, L> {
    private ArrayList<Duple<V, L>> data = new ArrayList<>();
    private ToDoubleBiFunction<V, V> distance;
    private int k;

    public Knn(int k, ToDoubleBiFunction<V, V> distance) {
        this.k = k;
        this.distance = distance;
    }

    @Override
    public L classify(V value) {
        //Finds Distance & K's neighbors
        PriorityQueue<Duple<Double , L>> nearestNeighbors =
                //Largest Dist First for because a Max-Heap
                new PriorityQueue<>(k, Comparator.comparing(Duple::getFirst, Comparator.reverseOrder()));
        //Calculates Distance
        for (Duple<V, L> trainingPoint : data) {
            double dist = distance.applyAsDouble(value, trainingPoint.getFirst());
            nearestNeighbors.offer(new Duple<>(dist, trainingPoint.getSecond()));
            //Keeps smallest K Distances
            if (nearestNeighbors.size() > k){
                nearestNeighbors.poll();
            }
        }

        Histogram<L> neighborLabels = new Histogram<>();
        for(Duple<Double, L> neighbor : nearestNeighbors){
            neighborLabels.bump(neighbor.getSecond());
        }
        return neighborLabels.getPluralityWinner();
    }

    @Override
    public void train(ArrayList<Duple<V, L>> training) {
        this.data = new ArrayList<>(training);
    }
}

package learning.decisiontree;

import core.Duple;
import learning.core.Histogram;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public class DTTrainer<V,L, F, FV extends Comparable<FV>> {
	private ArrayList<Duple<V,L>> baseData;
	private boolean restrictFeatures;
	private Function<ArrayList<Duple<V,L>>, ArrayList<Duple<F,FV>>> allFeatures;
	private BiFunction<V,F,FV> getFeatureValue;
	private Function<FV,FV> successor;
	
	public DTTrainer(ArrayList<Duple<V, L>> data, Function<ArrayList<Duple<V, L>>, ArrayList<Duple<F,FV>>> allFeatures,
					 boolean restrictFeatures, BiFunction<V,F,FV> getFeatureValue, Function<FV,FV> successor) {
		baseData = data;
		this.restrictFeatures = restrictFeatures;
		this.allFeatures = allFeatures;
		this.getFeatureValue = getFeatureValue;
		this.successor = successor;
	}
	
	public DTTrainer(ArrayList<Duple<V, L>> data, Function<ArrayList<Duple<V,L>>, ArrayList<Duple<F,FV>>> allFeatures,
					 BiFunction<V,F,FV> getFeatureValue, Function<FV,FV> successor) {
		this(data, allFeatures, false, getFeatureValue, successor);
	}

	// TODO: Call allFeatures.apply() to get the feature list. Then shuffle the list, retaining
	//  only targetNumber features. Should pass DTTest.testReduced().
	public static <V,L, F, FV  extends Comparable<FV>> ArrayList<Duple<F,FV>>
	reducedFeatures(ArrayList<Duple<V,L>> data, Function<ArrayList<Duple<V, L>>, ArrayList<Duple<F,FV>>> allFeatures,
					int targetNumber) {
		ArrayList<Duple<F, FV>> candidates = new ArrayList<>(allFeatures.apply(data));
		Collections.shuffle(candidates, new Random());
		return new ArrayList<>(candidates.subList(0, Math.min(targetNumber, candidates.size())));
    }
	
	public DecisionTree<V,L,F,FV> train() {
		return train(baseData);
	}

	public static <V,L> int numLabels(ArrayList<Duple<V,L>> data) {
		return data.stream().map(Duple::getSecond).collect(Collectors.toUnmodifiableSet()).size();
	}
	
	private DecisionTree<V,L,F,FV> train(ArrayList<Duple<V,L>> data) {
		if (data.isEmpty()) {
			throw new IllegalArgumentException("Empty training set");
		}
		// TODO: Implement the decision tree learning algorithm
		if (numLabels(data) == 1) {
			return new DTLeaf<>(data.get(0).getSecond());
			// TODO: Return a leaf node consisting of the only label in data
		}
		ArrayList<Duple<F, FV>> candidates;
		if (restrictFeatures) {
			int sqrt = (int) Math.sqrt(allFeatures.apply(baseData).size());
			candidates = reducedFeatures(data, allFeatures, sqrt);
		}
		else {
			candidates = allFeatures.apply(data);
		}

		double bestGain = -1.0;
		F bestFeature = null;
		FV bestValue = null;
		ArrayList<Duple<V, L>> bestLeft = null;
		ArrayList<Duple<V, L>> bestRight = null;

		for (Duple<F,FV> cand : candidates){
			F f = cand.getFirst();
			FV v = cand.getSecond();

			Duple<ArrayList<Duple<V, L>>, ArrayList<Duple<V, L>>> split = splitOn(data, f, v, getFeatureValue);

			ArrayList<Duple<V, L>> left = split.getFirst();
			ArrayList<Duple<V, L>> right = split.getSecond();

			if(left.isEmpty() || right.isEmpty()) continue;

			double g = gain(data, left, right);
			if(g > bestGain)
			{
				bestGain = g;
				bestFeature = f;
				bestValue = v;
				bestLeft = left;
				bestRight = right;
			}
		// TODO: Return an interior node.
			//  If restrictFeatures is false, call allFeatures.apply() to get a complete list
			//  of features and values, all of which you should consider when splitting.
			//  If restrictFeatures is true, call reducedFeatures() to get sqrt(# features)
			//  of possible features/values as candidates for the split. In either case,
			//  for each feature/value combination, use the splitOn() function to break the
			//  data into two parts. Then use gain() on each split to figure out which
			//  feature/value combination has the highest gain. Use that combination, as
			//  well as recursively created left and right nodes, to create the new
			//  interior node.
			//  Note: It is possible for the split to fail; that is, you can have a split
			//  in which one branch has zero elements. In this case, return a leaf node
			//  containing the most popular label in the branch that has elements

		}

		if(bestGain < 0) {
			return new DTLeaf<>(mostPopularLabelFrom(data));
		}

		DecisionTree<V, L, F, FV> leftTree = train(bestLeft);
		DecisionTree<V, L, F, FV> rightTree = train(bestRight);

		if(bestLeft.isEmpty()) {
			leftTree = new DTLeaf<>(mostPopularLabelFrom(bestRight));
		}
		if(bestRight.isEmpty()){
			rightTree = new DTLeaf<>(mostPopularLabelFrom(bestLeft));
		}

		return new DTInterior<>(bestFeature, bestValue, leftTree, rightTree, getFeatureValue, successor);
	}

	public static <V,L> L mostPopularLabelFrom(ArrayList<Duple<V, L>> data) {
		Histogram<L> h = new Histogram<>();
		for (Duple<V,L> datum: data) {
			h.bump(datum.getSecond());
		}
		return h.getPluralityWinner();
	}

	// TODO: Generates a new data set by sampling randomly with replacement. It should return
	//    an `ArrayList` that is the same length as `data`, where each element is selected randomly
	//    from `data`. Should pass `DTTest.testResample()`.
	public static <V,L> ArrayList<Duple<V,L>> resample(ArrayList<Duple<V,L>> data) {
		Random rnd = new Random();
		ArrayList<Duple<V, L>> sample = new ArrayList<>(data.size());
		for(int i = 0; i < data.size(); i++) {
			int idx = rnd.nextInt(data.size());
			sample.add(data.get(idx));
		}
		return sample;
	}

	public static <V,L> double getGini(ArrayList<Duple<V,L>> data) {
		// TODO: Calculate the Gini coefficient:
		//  For each label, calculate its portion of the whole (p_i).
		//  Use of a Histogram<L> for this purpose is recommended.
		//  Gini coefficient is 1 - sum(for all labels i, p_i^2)
		//  Should pass DTTest.testGini().
		Histogram<L> hist = new Histogram<>();
		for(Duple<V, L> d : data) {
			hist.bump(d.getSecond());
		}

		double sumSq = 1.0;
		int total = data.size();
		for(L label : hist) {
			double count = hist.getCountFor(label);
			double p = count / total;
			sumSq -= p*p;
		}
		return sumSq;
	}



	public static <V,L> double gain(ArrayList<Duple<V,L>> parent,
									ArrayList<Duple<V,L>> child1,
									ArrayList<Duple<V,L>> child2) {


		if (child1.isEmpty() || child2.isEmpty()) {
			return 0.0;
		}

		double gParent = getGini(parent);

		return gParent - (getGini(child1) + getGini(child2));

		// TODO: Calculate the gain of the split. Add the gini values for the children.
		//  Subtract that sum from the gini value for the parent. Should pass DTTest.testGain().
	}

	public static <V,L, F, FV  extends Comparable<FV>> Duple<ArrayList<Duple<V,L>>,ArrayList<Duple<V,L>>> splitOn
			(ArrayList<Duple<V,L>> data, F feature, FV featureValue, BiFunction<V,F,FV> getFeatureValue) {
		// TODO:
		//  Returns a duple of two new lists of training data.
		//  The first returned list should be everything from this set for which
		//  feature has a value less than or equal to featureValue. The second
		//  returned list should be everything else from this list.
		//  Should pass DTTest.testSplit().

		ArrayList<Duple<V, L>> left = new ArrayList<>();
		ArrayList<Duple<V, L>> right = new ArrayList<>();

		for(Duple<V, L> datum : data) {
			V valueContainer = datum.getFirst();
			FV fv = getFeatureValue.apply(valueContainer, feature);

			if(fv.compareTo(featureValue) <= 0) {
				left.add(datum);
			}
			else {
				right.add(datum);
			}
		}
		return new Duple<>(left, right);
	}
}

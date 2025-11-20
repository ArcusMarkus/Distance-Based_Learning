package learning.markov;

import learning.core.Histogram;

import java.util.*;

public class MarkovChain<L,S> {
    private LinkedHashMap<L, HashMap<Optional<S>, Histogram<S>>> label2symbol2symbol = new LinkedHashMap<>();

    public Set<L> allLabels() {return label2symbol2symbol.keySet();}

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (L language: label2symbol2symbol.keySet()) {
            sb.append(language).append('\n');
            for (Map.Entry<Optional<S>, Histogram<S>> entry: label2symbol2symbol.get(language).entrySet()) {
                sb.append("    ");
                sb.append(entry.getKey());
                sb.append(":");
                sb.append(entry.getValue().toString());
                sb.append('\n');
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    // Increase the count for the transition from prev to next.
    // Should pass SimpleMarkovTest.testCreateChains().
    public void count(Optional<S> prev, L label, S next) {
        label2symbol2symbol.putIfAbsent(label, new HashMap<>());
        HashMap<Optional<S>, Histogram<S>> prevToHist = label2symbol2symbol.get(label);

        prevToHist.putIfAbsent(prev, new Histogram<>());
        Histogram<S> histogram = prevToHist.get(prev);

        histogram.bumpBy(next, 1);
        // TODO: YOUR CODE HERE
    }

    // Returns P(sequence | label)
    // Should pass SimpleMarkovTest.testSourceProbabilities() and MajorMarkovTest.phraseTest()
    //
    // HINT: Be sure to add 1 to both the numerator and denominator when finding the probability of a
    // transition. This helps avoid sending the probability to zero.
    public double probability(ArrayList<S> sequence, L label) {
        // TODO: YOUR CODE HERE
        double logProb = 0.0;
        Optional<S> prev = Optional.empty();

        if(!label2symbol2symbol.containsKey(label) || sequence.isEmpty()) {
            return 0.0;
        }

        for (S next : sequence){
            HashMap<Optional<S>, Histogram<S>> prevToHist = label2symbol2symbol.get(label);
            Histogram<S> hist = prevToHist.getOrDefault(prev, new Histogram<>());

            int countNext = hist.getCountFor(next);
            int totalFromPrev = hist.getTotalCounts();

            double smoothProb = (countNext + 1.0) / (totalFromPrev + hist.getTotalCounts());
            logProb += Math.log(smoothProb);

            prev = Optional.of(next);
        }
        return Math.exp(logProb);
    }

    // Return a map from each label to P(label | sequence).
    // Should pass MajorMarkovTest.testSentenceDistributions()
    public LinkedHashMap<L,Double> labelDistribution(ArrayList<S> sequence) {
        // TODO: YOUR CODE HERE
        LinkedHashMap<L, Double> result = new LinkedHashMap<>();

        double sumLogProbs = Double.NEGATIVE_INFINITY;
        Map<L, Double> logProbs = new HashMap<>();

        for (L label : allLabels()) {
            double prob = probability(sequence, label);
            double logProb = (prob > 0) ? Math.log(prob) : Double.NEGATIVE_INFINITY;
            logProbs.put(label, logProb);
            sumLogProbs = logSumExp(sumLogProbs, logProb);
        }

        for (L label : allLabels()){
            double logProb = logProbs.get(label);
            double prob = (sumLogProbs == Double.NEGATIVE_INFINITY) ? 0.0 : Math.exp(logProb - sumLogProbs);
            result.put(label, prob);
        }
        return result;
    }

    private double logSumExp(double a, double b){
        if (a == Double.NEGATIVE_INFINITY) return b;
        if (b == Double.NEGATIVE_INFINITY) return a;
        if (a > b) {
            return a + Math.log(1.0 + Math.exp(b - a));
        } else {
            return b + Math.log(1.0 + Math.exp(a - b));
        }
    }

    // Calls labelDistribution(). Returns the label with highest probability.
    // Should pass MajorMarkovTest.bestChainTest()
    public L bestMatchingChain(ArrayList<S> sequence) {
        // TODO: YOUR CODE HERE
        if (allLabels().isEmpty()) return null;

        LinkedHashMap<L, Double> distribution = labelDistribution(sequence);

        L bestLabel = null;
        double bestProb = -1.0;

        for(Map.Entry<L, Double> entry : distribution.entrySet()){
            if(entry.getValue() > bestProb){
                bestProb = entry.getValue();
                bestLabel = entry.getKey();
            }
        }
        return bestLabel;
    }
}

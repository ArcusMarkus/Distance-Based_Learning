package robosim.reinforcement;

import java.util.Arrays;

public class QTable {
    private double[][] q;
    private int[][] visits;
    private int targetVisits;
    private double discount, rateConstant;
    private int lastState, lastAction;


    public double getLearningRate(int state, int action) {
        // TODO:
        //  Calculate the learning rate using this formula: 1/(1 + total visits for this (state, action) pair/rateConstant)
        //  Should pass QTableTest.testLearningRate().
        return 1.0 / (1.0 + visits[state][action] / rateConstant);
    }


    public int getBestAction(int state) {
        // TODO: Find the action for the given state that has the highest q value.
        //  Should pass QTableTest.testBestAction()
        int bestAction = 0;
        double maxQ = q[state][0];
        for (int action = 1; action < q[state].length; action++) {
            if(q[state][action] > maxQ) {
                maxQ = q[state][action];
                bestAction = action;
            }
        }
        return bestAction;
    }


    public boolean isExploring(int state) {
        // TODO: Returns true if any action for this state is below the target
        //  visits. Returns false otherwise.
        //  Should pass QTableTest.testIsExploring()
        for(int action = 0; action < visits[state].length; action++) {
            if(visits[state][action] < targetVisits) {
                return true;
            }
        }
        return false;
    }


    public int leastVisitedAction(int state) {
        // TODO: Returns the least visited action in state.
        //  Should pass QTableTest.testLeastVisitedAction()
        int least = 0;
        int min = visits[state][0];
        for(int a = 1; a < visits[state].length; a++) {
            if(visits[state][a] < min) {
                min = visits[state][a];
                least = a;
            }
        }
        return least;
    }


    public int senseActLearn(int newState, double reward) {
        // TODO:
        //  1. Calculate the update for the last state and action.
        //  2. Modify the q-value for the last state and action.
        //  3. Increase the visit count for the last state and action.
        //  4. Select the action for the new state.
        //     * If we are exploring, use the least visited action.
        //     * Otherwise, use the best action.
        //  5. Update the last state and action.
        //  6. Return the selected action.
        //  Should pass QTableTest.testSenseActLearn()
        //
        //  Q update formula:
        //    Q(s, a) = (1 - learningRate) * Q(s, a) + learningRate * (discount * maxa(Q(s', a)) + r(s))
        double lr = getLearningRate(lastState, lastAction);
        double maxQ = Double.NEGATIVE_INFINITY;


        for(int action = 0; action < q[newState].length; action++) {
            if(q[newState][action] > maxQ) {
                maxQ = q[newState][action];
            }
        }
        q[lastState][lastAction] = (1 - lr) * q[lastState][lastAction] + lr * (reward + discount * maxQ);
        visits[lastState][lastAction]++;
        int selectedAction;
        if (isExploring(newState)) {
            selectedAction = leastVisitedAction(newState);
        }
        else {
            selectedAction = getBestAction(newState);
        }
        lastState = newState;
        lastAction = selectedAction;
        return selectedAction;

    }

    public QTable(int states, int actions, int startState, int targetVisits, int rateConstant, double discount) {
        this.targetVisits = targetVisits;
        this.rateConstant = rateConstant;
        this.discount = discount;
        q = new double[states][actions];
        visits = new int[states][actions];
        lastState = startState;
        lastAction = 0;
    }

    private QTable() {}

    static QTable from(String s) {
        QTable result = new QTable();
        String[] lines = s.split("\n");
        String[] values1 = lines[0].split(";");
        result.targetVisits = Integer.parseInt(values1[0].split(":")[1]);
        result.discount = Double.parseDouble(values1[1].split(":")[1]);
        result.rateConstant = Double.parseDouble(values1[2].split(":")[1]);
        result.lastState = Integer.parseInt(values1[3].split(":")[1]);
        result.lastAction = Integer.parseInt(values1[4].split(":")[1]);

        boolean createdArrays = false;
        for (int i = 1; i < lines.length; i++) {
            String[] topSplit = lines[i].split(":");
            int state = Integer.parseInt(topSplit[0]);
            assert state == i - 1;
            String[] nextSplit = topSplit[1].split(",");
            for (int action = 0; action < nextSplit.length; action++) {
                String[] innerSplit = nextSplit[action].split("\\(");
                if (!createdArrays) {
                    int numStates = lines.length - 1;
                    int numActions = innerSplit.length;
                    result.q = new double[numStates][numActions];
                    result.visits = new int[numStates][numActions];
                    createdArrays = true;
                }
                result.q[state][action] = Double.parseDouble(innerSplit[0]);
                result.visits[state][action] = Integer.parseInt(innerSplit[1].substring(0, innerSplit[1].length() - 1));
            }
        }
        return result;
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("targetVisits:");
        result.append(targetVisits);
        result.append(";discount:");
        result.append(discount);
        result.append(";rateConstant:");
        result.append(rateConstant);
        result.append(";lastState:");
        result.append(lastState);
        result.append(";lastAction:");
        result.append(lastAction);
        result.append('\n');
        for (int state = 0; state < q.length; ++state) {
            result.append(state);
            result.append(':');
            for (int action = 0; action < q[state].length; action++) {
                result.append(q[state][action]);
                result.append('(');
                result.append(visits[state][action]);
                result.append(')');
                result.append(',');
            }
            result.deleteCharAt(result.length() - 1);
            result.append('\n');
        }
        return result.toString();
        //done
    }

    public double getQ(int state, int action) {
        return q[state][action];
    }

    public int getLastState() {
        return lastState;
    }

    public int getLastAction() {
        return lastAction;
    }
}
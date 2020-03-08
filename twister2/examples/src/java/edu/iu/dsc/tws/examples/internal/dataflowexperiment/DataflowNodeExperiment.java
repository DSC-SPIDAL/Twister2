package edu.iu.dsc.tws.examples.internal.dataflowexperiment;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import edu.iu.dsc.tws.api.JobConfig;
import edu.iu.dsc.tws.api.Twister2Job;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageTypes;
import edu.iu.dsc.tws.api.compute.IFunction;
import edu.iu.dsc.tws.api.compute.IMessage;
import edu.iu.dsc.tws.api.compute.TaskContext;
import edu.iu.dsc.tws.api.compute.executor.ExecutionPlan;
import edu.iu.dsc.tws.api.compute.graph.ComputeGraph;
import edu.iu.dsc.tws.api.compute.graph.OperationMode;
import edu.iu.dsc.tws.api.compute.nodes.BaseCompute;
import edu.iu.dsc.tws.api.compute.nodes.BaseSource;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.config.SchedulerContext;
import edu.iu.dsc.tws.data.utils.DataObjectConstants;
import edu.iu.dsc.tws.rsched.core.ResourceAllocator;
import edu.iu.dsc.tws.rsched.job.Twister2Submitter;
import edu.iu.dsc.tws.task.impl.ComputeConnection;
import edu.iu.dsc.tws.task.impl.ComputeGraphBuilder;
import edu.iu.dsc.tws.task.impl.TaskWorker;

public class DataflowNodeExperiment extends TaskWorker {

  private static final Logger LOG = Logger.getLogger(DataflowNodeExperiment.class.getName());

  @SuppressWarnings("unchecked")
  @Override
  public void execute() {
    LOG.log(Level.INFO, "Task worker starting: " + workerId);

    SourceTask sourceTask = new SourceTask();
    ReduceTask reduceTask = new ReduceTask();
    ComputeTask computeTask = new ComputeTask();

    ComputeGraphBuilder builder = ComputeGraphBuilder.newBuilder(config);
    DataflowJobParameters dataflowJobParameters = DataflowJobParameters.build(config);

    int parallel = dataflowJobParameters.getParallelismValue();
    int iter = dataflowJobParameters.getIterations();

    builder.addSource("source", sourceTask, parallel);
    ComputeConnection computeConnection = builder.addCompute("compute", computeTask, parallel);
    ComputeConnection rc = builder.addCompute("sink", reduceTask, parallel);

    computeConnection.direct("source")
        .viaEdge("direct")
        .withDataType(MessageTypes.OBJECT);

    rc.allreduce("compute")
        .viaEdge("all-reduce")
        .withReductionFunction(new Aggregator())
        .withDataType(MessageTypes.OBJECT);

    builder.setMode(OperationMode.BATCH);
    ComputeGraph graph = builder.build();
    ExecutionPlan plan = taskExecutor.plan(graph);
    long startTime = System.currentTimeMillis();
    for (int i = 0; i < iter; i++) {
      taskExecutor.execute(graph, plan);
      LOG.info("Completed Iteration:" + i);
    }
    long stopTime = System.currentTimeMillis();
    long executionTime = stopTime - startTime;
    LOG.info("Total Execution Time to Complete Dataflow Node Experiment"
        + "\t" + executionTime + "(in milliseconds)");
  }

  private static class SourceTask extends BaseSource {
    private static final long serialVersionUID = -254264120110286748L;

    private double[] datapoints = null;
    private int numPoints = 0;

    @Override
    public void execute() {
      int seedValue = 100;
      datapoints = new double[numPoints];
      Random r = new Random(seedValue);
      for (int i = 0; i < numPoints; i++) {
        double randomValue = r.nextDouble();
        datapoints[i] = randomValue;
      }
      context.writeEnd("direct", datapoints);
    }

    public void prepare(Config cfg, TaskContext context) {
      super.prepare(cfg, context);
      numPoints = Integer.parseInt(cfg.getStringValue(DataObjectConstants.DSIZE));
    }
  }

  private static class ComputeTask extends BaseCompute {
    private static final long serialVersionUID = -254264120110286748L;

    private int count = 0;

    @Override
    public boolean execute(IMessage message) {
      if (message.getContent() instanceof Iterator) {
        Iterator it = (Iterator) message.getContent();
        while (it.hasNext()) {
          count += 1;
          context.write("all-reduce", it.next());
        }
      }
      context.end("all-reduce");
      return true;
    }
  }

  private static class ReduceTask extends BaseCompute {
    private static final long serialVersionUID = -5190777711234234L;

    private double[] datapoints;

    @Override
    public boolean execute(IMessage message) {
      datapoints = (double[]) message.getContent();
      return true;
    }
  }

  public class Aggregator implements IFunction {
    private static final long serialVersionUID = -254264120110286748L;

    @Override
    public Object onMessage(Object object1, Object object2) throws ArrayIndexOutOfBoundsException {

      double[] object11 = (double[]) object1;
      double[] object21 = (double[]) object2;

      double[] object31 = new double[object11.length];

      for (int j = 0; j < object11.length; j++) {
        double newVal = object11[j] + object21[j];
        object31[j] = newVal;
      }
      return object31;
    }
  }

  public class Aggregator1 implements IFunction {
    private static final long serialVersionUID = -254264120110286748L;

    @Override
    public Object onMessage(Object object1, Object object2) throws ArrayIndexOutOfBoundsException {

      double[] object11 = (double[]) object1;
      double[] object21 = (double[]) object2;

      double[] object31 = new double[object11.length];

      for (int j = 0; j < object11.length; j++) {
        double newVal = object11[j] + object21[j];
        object31[j] = newVal;
      }
      return object31;
    }
  }

  public static void main(String[] args) throws ParseException {
    // first load the configurations from command line and config files
    Config config = ResourceAllocator.loadConfig(new HashMap<>());

    // build JobConfig
    HashMap<String, Object> configurations = new HashMap<>();
    configurations.put(SchedulerContext.THREADS_PER_WORKER, 8);

    Options options = new Options();
    options.addOption(DataObjectConstants.ARGS_ITERATIONS, true, "iter");
    options.addOption(DataObjectConstants.PARALLELISM_VALUE, true, "parallelism");
    options.addOption(DataObjectConstants.WORKERS, true, "workers");
    options.addOption(DataObjectConstants.DSIZE, true, "dsize");

    CommandLineParser commandLineParser = new DefaultParser();
    CommandLine cmd = commandLineParser.parse(options, args);
    int workers = Integer.parseInt(cmd.getOptionValue(DataObjectConstants.WORKERS));
    int parallelismValue = Integer.parseInt(cmd.getOptionValue(
        DataObjectConstants.PARALLELISM_VALUE));
    int iterations = Integer.parseInt(cmd.getOptionValue(
        DataObjectConstants.ARGS_ITERATIONS));
    int dsize = Integer.parseInt(cmd.getOptionValue(DataObjectConstants.DSIZE));

    // build JobConfig
    JobConfig jobConfig = new JobConfig();
    jobConfig.put(DataObjectConstants.DSIZE, Integer.toString(dsize));
    jobConfig.put(DataObjectConstants.WORKERS, Integer.toString(workers));
    jobConfig.put(DataObjectConstants.PARALLELISM_VALUE, Integer.toString(parallelismValue));
    jobConfig.put(DataObjectConstants.ARGS_ITERATIONS, Integer.toString(iterations));
    jobConfig.putAll(configurations);

    Twister2Job.Twister2JobBuilder jobBuilder = Twister2Job.newBuilder();
    jobBuilder.setJobName("Experiment1");
    jobBuilder.setWorkerClass(DataflowNodeExperiment.class.getName());
    jobBuilder.addComputeResource(2, 512, 1.0, workers);
    jobBuilder.setConfig(jobConfig);

    // now submit the job
    Twister2Submitter.submitJob(jobBuilder.build(), config);
  }
}



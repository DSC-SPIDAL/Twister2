package edu.iu.dsc.tws.python;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import edu.iu.dsc.tws.python.config.ComputeResource;

public class BootstrapPoint extends EntryPoint {

  private Map<String, Object> configs = new HashMap<>();
  private List<ComputeResource> computeResources = new ArrayList<>();
  private String jobName = "python-job-" + UUID.randomUUID().toString();

  public void addConfig(String key, Object value) {
    this.configs.put(key, value);
  }

  public void setJobName(String jobName) {
    if (jobName != null) {
      this.jobName = jobName;
    }
  }

  public void createComputeResource(Number cpu, Number ram, Number instances) {
    this.computeResources.add(new ComputeResource(cpu.doubleValue(), ram.intValue(),
        instances.intValue()));
  }

  public String getJobName() {
    return jobName;
  }

  public List<ComputeResource> getComputeResources() {
    return computeResources;
  }

  public Map<String, Object> getConfigs() {
    return configs;
  }
}

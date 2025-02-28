import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"

const TPU_CONFIGS = {
  v6e: {
    hbm: 32,
    pricePerHour: 2.7,
    topologies: ["1x1", "2x2", "2x4", "2x4", "4x4", "4x8", "8x8", "8x16", "16x16"],
  },
  v5p: {
    hbm: 96,
    pricePerHour: 4.2,
    topologies: [
      "2x2x1",
      "2x2x2",
      "2x4x4",
      "4x4x4",
      "4x4x8",
      "4x8x8",
      "8x8x8",
      "8x8x16",
      "8x16x16",
      "16x16x16",
      "16x16x24",
    ],
  },
  v5e: {
    hbm: 16,
    pricePerHour: 1.2,
    topologies: ["1x1", "2x2", "2x4", "4x4", "4x8", "8x8", "8x16", "16x16"],
  },
  v4: {
    hbm: 32,
    pricePerHour: 3.22,
    topologies: [
      "2x2x1",
      "2x2x2",
      "2x2x4",
      "2x4x4",
      "4x4x4",
      "4x4x8",
      "4x8x8",
      "4x4x16",
      "8x8x8",
      "8x8x12",
      "8x8x16",
      "4x16x16",
      "8x16x16",
    ],
  },
}

export default function TPUCalculator() {
  const [modelSize, setModelSize] = useState(2)
  const [finetuningType, setFinetuningType] = useState("full")
  const [partialPercentage, setPartialPercentage] = useState(5)
  const [batchSize, setBatchSize] = useState(32)
  const [sequenceLength, setSequenceLength] = useState(512)
  const [result, setResult] = useState(null)

  const calculateHBM = () => {
    // Model Parameters calculation
    const modelParameters = modelSize * 4

    // Optimizer State calculation
    const fullOptimizer = modelParameters * 3
    const optimizerState = finetuningType === "full" ? fullOptimizer : fullOptimizer * (partialPercentage / 100)

    // Intermediate Buffer calculation
    const intermediateBuffer = 10 + (batchSize * sequenceLength) / 2048

    // Total HBM calculation
    const totalHBM = modelParameters + optimizerState + intermediateBuffer

    // Calculate recommendations for each TPU type
    const recommendations = Object.entries(TPU_CONFIGS).map(([tpuType, config]) => {
      const devicesNeeded = Math.ceil(totalHBM / config.hbm)

      // Find the smallest topology that can accommodate the required devices
      let selectedTopology = config.topologies[0]
      for (const topology of config.topologies) {
        const devices = topology.split("x").reduce((acc, val) => acc * Number.parseInt(val), 1)
        if (devices >= devicesNeeded) {
          selectedTopology = topology
          break
        }
      }

      const actualDevices = selectedTopology.split("x").reduce((acc, val) => acc * Number.parseInt(val), 1)
      const costPerHour = actualDevices * config.pricePerHour

      return {
        tpuType,
        hbmPerDevice: config.hbm,
        devicesNeeded,
        recommendedTopology: selectedTopology,
        actualDevices,
        costPerHour,
      }
    })

    setResult({
      modelParameters,
      optimizerState,
      intermediateBuffer,
      totalHBM,
      recommendations,
    })
  }

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>TPU HBM Calculator</CardTitle>
        <CardDescription>Calculate HBM requirements, TPU topologies, and costs for your model</CardDescription>
      </CardHeader>
      <CardContent>
        <Accordion type="single" collapsible className="mb-6">
          <AccordionItem value="formulas">
            <AccordionTrigger>View Formulas Used</AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4 p-4 bg-muted rounded-lg">
                <div>
                  <h4 className="font-semibold">Total HBM Required</h4>
                  <p className="text-sm">Model Parameters + Optimizer State + Buffer for intermediates</p>
                </div>
                <div>
                  <h4 className="font-semibold">Model Parameters</h4>
                  <p className="text-sm">Required Memory = (Model Size in Billions) × 4GB</p>
                </div>
                <div>
                  <h4 className="font-semibold">Optimizer State</h4>
                  <p className="text-sm">Full Fine-tuning: 3 × Model Parameter Memory</p>
                  <p className="text-sm">Partial Fine-tuning: (% trainable parameters) × Full Optimizer Memory</p>
                </div>
                <div>
                  <h4 className="font-semibold">Buffer for intermediates</h4>
                  <p className="text-sm">10GB + (batch_size × sequence_length / 2048)</p>
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        <form
          onSubmit={(e) => {
            e.preventDefault()
            calculateHBM()
          }}
          className="space-y-4"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="modelSize">Model Size (billions of parameters)</Label>
              <Input
                id="modelSize"
                type="number"
                value={modelSize}
                onChange={(e) => setModelSize(Number.parseFloat(e.target.value))}
                min="0.1"
                step="0.1"
              />
            </div>
            <div>
              <Label htmlFor="finetuningType">Fine-tuning Type</Label>
              <Select value={finetuningType} onValueChange={setFinetuningType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="full">Full Fine-tuning</SelectItem>
                  <SelectItem value="partial">Partial Fine-tuning</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {finetuningType === "partial" && (
              <div>
                <Label htmlFor="partialPercentage">Partial Fine-tuning Percentage</Label>
                <Input
                  id="partialPercentage"
                  type="number"
                  value={partialPercentage}
                  onChange={(e) => setPartialPercentage(Number.parseFloat(e.target.value))}
                  min="0"
                  max="100"
                />
              </div>
            )}
            <div>
              <Label htmlFor="batchSize">Batch Size</Label>
              <Input
                id="batchSize"
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(Number.parseInt(e.target.value))}
                min="1"
              />
            </div>
            <div>
              <Label htmlFor="sequenceLength">Sequence Length</Label>
              <Input
                id="sequenceLength"
                type="number"
                value={sequenceLength}
                onChange={(e) => setSequenceLength(Number.parseInt(e.target.value))}
                min="1"
              />
            </div>
          </div>
          <Button type="submit">Calculate</Button>
        </form>

        {result && (
          <div className="mt-8 space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card>
                <CardHeader className="p-4">
                  <CardTitle className="text-lg">Model Parameters</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{result.modelParameters.toFixed(2)} GB</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="p-4">
                  <CardTitle className="text-lg">Optimizer State</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{result.optimizerState.toFixed(2)} GB</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="p-4">
                  <CardTitle className="text-lg">Intermediate Buffer</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{result.intermediateBuffer.toFixed(2)} GB</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="p-4">
                  <CardTitle className="text-lg">Total HBM Required</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{result.totalHBM.toFixed(2)} GB</p>
                </CardContent>
              </Card>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-4">TPU Recommendations</h3>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>TPU Type</TableHead>
                    <TableHead>HBM/Device</TableHead>
                    <TableHead>Recommended Topology</TableHead>
                    <TableHead>Total Devices</TableHead>
                    <TableHead>Cost per Hour</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {result.recommendations.map((rec) => (
                    <TableRow key={rec.tpuType}>
                      <TableCell className="font-medium">TPU {rec.tpuType}</TableCell>
                      <TableCell>{rec.hbmPerDevice} GB</TableCell>
                      <TableCell>{rec.recommendedTopology}</TableCell>
                      <TableCell>{rec.actualDevices}</TableCell>
                      <TableCell>${rec.costPerHour.toFixed(2)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}


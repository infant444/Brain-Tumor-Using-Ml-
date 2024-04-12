import { Component, OnInit ,Input} from '@angular/core';
import { Patient } from '../../../model/patient';
import { BraintumorService } from '../../../Services/brain/braintumor.service';
import { ActivatedRoute, Router } from '@angular/router';

@Component({
  selector: 'app-brain',
  templateUrl: './brain.component.html',
  styleUrl: './brain.component.css'
})
export class BrainComponent implements OnInit {
  @Input()
  report!:Patient;

  tc="red";
  constructor(private router:Router,private activaterout:ActivatedRoute,private braintumor:BraintumorService){}
  ngOnInit(): void {
    this.activaterout.params.subscribe((params)=>{
      if(params.reportid){
        this.braintumor.getsinglereports(params.reportid).subscribe((report)=>{
          this.report=report;
        })
      }
    })
    if(this.report.brain_tumor){
  this.tc="red";
    }
    else{
  this.tc="green";

    }
  }
  open(){
    this.router.navigateByUrl("/braintumor/"+this.report.id);

  }
}

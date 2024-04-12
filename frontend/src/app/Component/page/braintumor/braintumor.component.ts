import { Component, OnInit } from '@angular/core';
import { BraintumorService } from '../../../Services/brain/braintumor.service';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { ToastrService } from 'ngx-toastr';
import { DatePipe } from '@angular/common';
import { UserService } from '../../../Services/user/user.service';
import { User } from '../../../model/user';
import { ActivatedRoute, Router } from '@angular/router';
import { Patient } from '../../../model/patient';

@Component({
  selector: 'app-braintumor',
  templateUrl: './braintumor.component.html',
  styleUrl: './braintumor.component.css'
})
export class BraintumorComponent implements OnInit {

  braintest!: FormGroup;
  issumit = false;
  images!: any;
  res!: string;
  resimg!: any;
  imageurl!: string;
  pdfurl: string = "";
  selctedsym: any[] = [];
  datePipe = new DatePipe('en-US');
  currentdate=new Date()
  datemax=this.datePipe.transform(this.currentdate, 'yyyy-MM-dd');
  user!:User;
  report!:Patient;
  symproms = ['Headaches','Vomiting','Seizures','Changes_in_vision','Weakness','Difficulty_speaking','Changes_in_personality','Memory_problems','Coordination_difficulties','Changes_in_hearing','Difficulty_swallowing','Changes_in_sensation','Fatigue','Cognitive impairment','Unexplained_weight_loss','Unexplained_weight_gain',

  ]
  constructor(private braintumor: BraintumorService, private Formbuilder: FormBuilder, private toasroe: ToastrService,private userservices:UserService,private activarerouter:ActivatedRoute,private router:Router) {
    this.user=userservices.currentUser;
  }
  ngOnInit(): void {
    this.braintest = this.Formbuilder.group({
      name: ['', Validators.required],
      dob: ['', Validators.required],
      gender: ['male', Validators.required],
      data: ['', Validators.required],
      Symptoms: ['',[]]
    });
    this.activarerouter.params.subscribe((params)=>{
      if(params.reportid){
        // this.braintumor.getreport(params.reportid).subscribe((x) => {
        //       this.router.navigateByUrl("/")
        //     })
        this.braintumor.getreport(params.reportid).subscribe((data) => {
          const blob = new Blob([data], { type: 'application/pdf' });
          const url = window.URL.createObjectURL(blob);
          const a=document.createElement('a');
          a.href = url;
          a.download='report.pdf'
          a.click();
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);

              this.router.navigateByUrl("/")
            })
      }
    })
  }
  get Fc() {
    return this.braintest.controls;
  }
  change(event: any) {

    // console.log(event.target.files[0]);
    if (event.target.files.length > 0) {
      this.images = event.target.files[0];
    }
  }

  submit() {
    this.issumit = true
    if (this.braintest.invalid) {

      console.log(this.braintest.getError)
      return
    }
    const formdata = new FormData();
    const formvalue=this.braintest.value;
   const DOB = new Date(formvalue.dob)
    const formDOB = this.datePipe.transform(DOB, 'dd/MM/yyyy')?.toString();
    var age=this.currentdate.getFullYear()-DOB.getFullYear();

    if(this.currentdate.getMonth()<DOB.getMonth()){
      age--;
    }
    else if(this.currentdate.getMonth()==DOB.getMonth()){
      if(this.currentdate.getDate()<DOB.getDate()){
        age--
      }
    }
    formdata.append("image", this.images);
    formdata.append("name",formvalue.name);
    formdata.append("dob",formDOB as string);
    formdata.append("age",age.toString());
    formdata.append("gender",formvalue.gender);
    formdata.append("symptoms",formvalue.Symptoms);
    formdata.append("user",this.user.id);

    this.braintumor.testbrain(formdata).subscribe(
      {
        next: (x) => {
          this.report=x;

        }
      }
    )
  }
  // z() {
  //   this.braintumor.sample().subscribe((x) => {
  //     console.log(x)
  //   })
  // }

}




